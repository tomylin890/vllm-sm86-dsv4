## 疑難排解

以下每一條都是這個專案實際撞過的，症狀在前。

排查前先從啟動日誌抓這三處，後面幾乎每一條都會用到：

```
DeepseekV4 serving profile PROFILE-CACHE (compressor-state ring: off, prefix caching: on,
max_num_batched_tokens: 768), scheduler_block_size=1024, hash_block_size=4,
num_gpu_blocks=..., sliding_window_groups=...
Available KV cache memory: X GiB
GPU KV cache size: N tokens
```

第一行來自`compressor.py`的profile判定，直接告訴你這次開的是PROFILE-P8還是PROFILE-CACHE，不必去猜環境變數有沒有生效；它後半段那幾個數字是拿來斷言部署矩陣的（B2那類block size問題會在這裡先露出馬腳）。第二行是**物理**可用量。第三行是套用`--num-gpu-blocks-override`之後的池子大小。

### 引擎不回應，日誌重複出現「No available shared memory broadcast block found in 60 seconds」

這句是`shm_broadcast.py`在自旋等待時每隔`VLLM_RINGBUFFER_WARNING_INTERVAL`（預設60）秒印一次的INFO。注意訊息裡的「60 seconds」印的是那個間隔常數，不是已經等了多久——等十分鐘它照樣寫60，看到這行不要以為只卡了一分鐘。

它的意思是「某個peer沒有推進」。開機階段編譯或量化權重時出現是正常的；服務中出現代表有一個worker已經死了或卡死，而其他worker還在等它。

往上翻同一份日誌，找這兩行：

```
WorkerProc hit an exception.
Worker proc <name> died unexpectedly (exit code: <n>), shutting down executor.
```

第一行是worker自己的traceback，那才是真因（在這台機器上最常見的是長請求prefill中途OOM）。第二行是父行程的監控執行緒發現它死掉。上游vLLM原本只讓`output_rank`回報例外，非回報rank的例外會被吞掉，於是唯一的表現就是引擎安靜地卡住；本分支已改成非回報rank重新拋出（見「對上游的發現」），所以現在traceback一定印得出來，但你必須自己往上找。

處置：`pkill`（下一條）之後重開。這個狀態不會自癒。

另外，ssh在這種時候連不上不是獨立事件。八個卡死的worker仍在NCCL自旋，5700X的16個執行緒被吃滿，sshd會反應不過來甚至連不上。**ssh失敗是症狀不是原因**——先別急著判機器掛了或去斷電，從已經連著的session下手，或等自旋的行程被殺掉。

### worker死了之後引擎沒有自己收乾淨

即使監控執行緒印出`died unexpectedly ... shutting down executor`，teardown本身還是會卡在shm broadcast的等待上，行程留著、VRAM不放。這是已知的缺口，還沒修。

所以清理不能只靠`Ctrl-C`：

```bash
pkill -9 -f "VLLM::"
pkill -9 -f "vllm[ ]serve"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

三件事都必要，理由各不相同：

`VLLM::`那條不能省。worker的行程標題是`VLLM::Worker_PP0_TP0_DCP0`這種格式（`set_process_title`會加上`VLLM::`前綴），`pkill -f "vllm serve"`只殺得到launcher，worker會活下來，每張卡繼續佔著二十幾GB，下一輪啟動在`set_device_index`就OOM。

`"vllm[ ]serve"`的方括號不是筆誤。`pkill -f`比對的是完整命令列，如果你用`ssh host 'pkill -f "vllm serve"'`，遠端那個`bash -c`的命令列裡自己就含有`vllm serve`這串字，於是它會殺掉自己、ssh當場斷線，而真正的引擎還活著。寫成`vllm[ ]serve`就解開了：這個regex仍然匹配得到真正的`vllm serve`命令列，但它自己的命令列上是字面的`vllm[ ]serve`，方括號沒被regex匹配到，自匹配就破了。同理，**kill和launch不要放在同一條ssh指令裡**。

VRAM檢查是收尾條件，不是禮貌。`pkill -9`回來之後行程未必已經釋放記憶體，要輪詢到每張卡都掉回500MiB以下才能開下一輪，否則下一次boot會用一個被污染的可用量去做memory profiling。順序是：`pkill -9`、輪詢`nvidia-smi --query-gpu=memory.used`直到每張卡低於500MiB、才開下一輪。腳本裡寫的是不帶方括號的`vllm serve`，因為它是在機器上直接執行、自己的命令列不含那串字；只有從`ssh host '...'`裡下手時才需要上面那個方括號寫法。

### 開機時ncclUnhandledCudaError出現在PP的broadcast，但grep OutOfMemoryError一個都沒有

這是`--num-gpu-blocks-override`開太大。實測override到1600時必現，1200確定安全，天花板在兩者之間。

機制：NCCL的通訊緩衝是第一次使用通訊器時才惰性配置的，那個時間點在KV池子配置完之後。池子把VRAM吃到只剩零頭，NCCL要不到自己那份，回報的是自己的錯誤碼而不是torch的`OutOfMemoryError`。所以這是一個記憶體不足問題，卻不會出現在任何一個你習慣grep的關鍵字裡。

判別方式就是這個組合：

```bash
grep -c OutOfMemoryError <log>   # 0
grep -n ncclUnhandledCudaError <log>   # 命中，且上下文在PP的broadcast
```

看到這個組合就往下調override，不要往上調`--gpu-memory-utilization`——util在override生效時完全不影響實際用量。

### 開機被拒，訊息說available只剩0.18 GiB，但看起來明明還有

錯誤長這樣：

```
To serve at least one request with the model's max seq len (262144), (0.55 GiB KV cache is
needed, which is larger than the available KV cache memory (0.18 GiB).
```

這裡的`available`是**套用override之後的有效容量**，不是物理可用量。我實際踩過的一次：物理是1.74 GiB，override設400，訊息就寫0.18，看起來像記憶體不夠，實際上是我自己把池子設得太小。

要看物理量請找這行：

```
Available KV cache memory: X GiB
```

兩個數字對不上就是override的問題，改override；兩個數字接近才是真的不夠，那時候能動的槓桿只有降F、砍暫態預算、或降`--max-model-len`（理由見「24GB卡的記憶體陷阱」）。

我在這裡誤判過兩次，兩次都白花了一輪boot。

### 記憶體莫名少了兩百多MB，沒有任何警告

檢查`PYTORCH_CUDA_ALLOC_CONF`裡有沒有同時出現`expandable_segments:True`和`max_split_size_mb`。兩者併用時expandable會被靜默停用，PyTorch不印警告、日誌裡也沒有任何痕跡，唯一的表現就是碎片變多、可用量比上一次少一截，然後在某個長請求上OOM。

沒有log line可以grep，只能直接讀行程的環境：

```bash
tr '\0' '\n' < /proc/$(pgrep -f "vllm[ ]serve" | head -1)/environ | grep PYTORCH_CUDA_ALLOC_CONF
```

正確的值就是單獨一項：

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 重開機後第一次請求慢一倍，跑第二次就正常

這是Triton JIT，不是硬體問題也不是配置退化。

具體數字：同一個200k needle，冷的那次72秒、暖了之後44秒；decode冷的22 tok/s、暖的49。差不多就是整整一半，大到會讓人以為配置壞了。

原因是編譯分桶落在「一個請求消耗了幾個prefill chunk」這個維度上，每個新的chunk數量桶要編一次，每次7-11.5秒，而且是per-process。同一個桶內共享，所以10668和10752只差0.87秒；跨桶就重付。

引擎內建的暖機（`VLLM_DSV4_WARMUP`，預設開）會在boot時走一遍chunk數階梯，日誌裡確認這行：

```
DSV4 SM8x JIT warmup: mixed dummy runs at token sizes [...]
```

但它蓋不完。實測boot之後仍然是每個新長度付一次學費，所以部署SOP是開機後從HTTP層再打一輪暖機請求（5個代表長度各一發，例如2048/16384/65536/131072/204800）再掛流量；用streaming發，連線卡住才會變成timeout而不是無限等。

自己量測時掃兩輪取第二輪。第一輪的數字不是真實性能，把它寫進報告只會誤導自己。

### 掃到190k以上時OOM僵死

暫態預算疊在一起爆掉。兩個旋鈕直接砍：

```
VLLM_DSV4_DELTA_GATHER_BUDGET_MB=256
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
```

代價實測在128k以下約1%（5194對照未修剪的5231 tok/s），換來253952那一格從必死變成穩定通過。這兩個值是暫態工作區的上限，不是池子，砍它們不會動到`GPU KV cache size`。

### 兩條請求併發，decode掉到5 tok/s

先確認`--long-prefill-token-threshold`有沒有大於`--max-num-batched-tokens`的一半。相等時一條長prefill會吃光每步的token預算，第二條請求要等第一條prefill完才進得來，於是變成「一條decode時另一條在prefill」，而被prefill佔滿的排程步只讓decode前進一個token——實測4.2-4.8 tok/s，對照單流的50。

這與記憶體無關（發生時kv使用率只有49%），也與F無關（F=512配override 1200症狀完全相同）。

```bash
curl -s http://127.0.0.1:8000/metrics | grep vllm:kv_cache_usage_perc
```

如果這個值不高但第二條請求就是進不來，那是入場預留而不是實際用量把它擋在外面，看usage會誤判。完整的機制與長度分界點見「併發：262144下只能單流，這是結構性的」。

### 兩次跑同一個prompt，輸出不一樣

這是預期行為，不是缺陷，而且與prefix caching無關。根因在上游的top-k平手競態，詳見「不保證跨次可重現」。做A/B比對時用`VLLM_SM86_DET_TOPK=1`把這個變數固定住，但不要開在生產環境（很慢，而且它本身也不是「正確答案」）。

### 引擎在init就AssertionError，訊息關於block size整除

```
Each KV cache group's real block_size must be divisible by hash_block_size.
block_sizes=[...], hash_block_size=...
```

`hash_block_size`預設是各群組block size的GCD，而dcp_exempt的群組在GCD那一側必須用**未乘dcp**的尺寸。這個配置下manager的真實尺寸是`[1024, 64, 4, 8]`，GCD是4，斷言成立；一旦exempt群組在算GCD時被誤乘了dcp，GCD就變成16，`4 % 16`當場炸。分支內已經把scheduler側與manager側的dcp_exempt判定統一（就是P11的B2），正常配置不該撞到這裡。

所以撞到它就代表某個尺寸真的被改動了——`--decode-context-parallel-size`、`--block-size`、或壓縮比。`--prefix-match-unit`可以手動指定這個值，但它只接受能整除**每一個**群組block size的數，在這裡上限就是4（也就是預設值；1和2合法但更細，只是讓block hash算更多次）。沒有往上繞的空間，訊息裡那組`block_sizes`才是要看的東西。

### 開機被拒，訊息提到PROFILE-P8與PROFILE-CACHE互斥

你同時設了`VLLM_DSV4_COMPRESSOR_WINDOWED=1`和`--enable-prefix-caching`。這是刻意的硬性拒絕，不會靜默降級——兩種擺法的KV佔用差以GB計，靜默選一邊只會在很久之後以入場拒絕或OOM的形式回來。照訊息挑一組配置，理由見「兩組配置怎麼選」。

有一處要注意：這則訊息裡建議PROFILE-CACHE把`max_num_batched_tokens`壓到512，那是當初寫這段守衛時的保守值。後來實測出來的上限是768（見「兩組配置怎麼選」的記憶體帳），以768為準。訊息本身還沒改。
