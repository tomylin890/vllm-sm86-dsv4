# 8卡3090跑DeepSeek-V4-Flash-0731

*[English](README.md)*

284B參數的MoE（13B啟動），八張消費級24GB卡，262144上下文，prefill峰值5730 tok/s，decode全上下文平坦50 tok/s，prefix caching可用。

這是vLLM的分支，核心是幫Ampere（SM86）補上decode context parallelism。上游的DCP依賴Hopper以上的核心，而DSV4的混合KV佈局（壓縮KV＋稀疏索引器＋滑動視窗）本來就與vLLM的上下文平行機制不相容，兩個問題疊在一起，所以在消費級卡上一直沒有可用的長上下文方案。

這是個人專案，在自有機器上完成，不提供支援，也不保證在其他環境能複現。

## 文件

這份README是總覽與實測結果。細節分在四份文件裡：

- [安裝與啟動](docs-dsv4/install.zh-TW.md)——環境需求、以預編譯wheel安裝這個分支的方法、flash-mla的sm86建置、兩組配置的完整啟動指令（每個flag一行說明）、開機後的暖機程序
- [架構](docs-dsv4/architecture.zh-TW.md)——DSV4的KV佈局究竟長什麼樣、為什麼vLLM的上下文平行不能直接用、這個分支各階段做了什麼。想接手改code的話從這裡開始
- [疑難排解](docs-dsv4/troubleshooting.zh-TW.md)——症狀導向。每一條都是這個專案實際撞過的，附上辨識用的log行或指令


## 為什麼難

模型權重fp8就佔156GB，八張卡合計192GB，剩下的空間要放262K的KV cache，每張卡實際可用不到1GB。在這個尺度下任何浪費都是致命的。

消費級卡沒有NVLink。我這台是兩顆PCIe switch串接、每顆掛四張卡，跨switch的頻寬是明確的瓶頸——TP8在52-62k上下文處會撞牆，原因就是張量平行的集合通訊跨了switch。最終配置是TP4+PP2+dcp4：TP只在單一switch內完成，PP跨switch（通訊量小一個數量級），DCP負責把壓縮KV分片。

其他拓樸下這個配置不一定最佳，但「TP群組不要跨switch」這條應該是通用的。

## 實測數據

單流序列提交，溫度0，暖機後量測（暖機的必要性見下文）。

### 配置一：PROFILE-P8，最高吞吐，無prefix caching

| | |
|---|---|
| 最大上下文 | 262144 |
| prefill峰值 | 5730 tok/s（@21k） |
| prefill平均 | 5248 tok/s（512→123k全掃） |
| decode | 49.9-51.2 tok/s，全上下文平坦 |
| 併發2聚合prefill | 5425 tok/s |
| 併發2聚合decode | 63.5-80.0 tok/s |
| 200k needle檢索 | 44秒通過 |

decode平坦這一點值得說明。一般實作的decode速度會隨上下文變長而衰減，這裡從短上下文到254k幾乎不動，原因是換上了flash-mla的sm86稀疏核心後，decode成本只跟選中的512個token有關，與序列總長度無關。

### 配置二：PROFILE-CACHE，開啟prefix caching

| 上下文 | 冷啟TTFT | 命中TTFT | 加速 |
|---|---|---|---|
| 32768 | 8.5s | 0.41s | 21× |
| 65536 | 17.3s | 0.48s | 36× |
| 131072 | 34.6s | 0.66s | 52× |
| 200704 | 56.2s | 0.86s | 65× |
| 253952 | 73.8s | 0.99s | 75× |

上表的冷啟數字量於F=512。F可以提高到768（見下一節的記憶體帳），冷啟prefill因此回升：

| 上下文 | F=512 | F=768 | 差 |
|---|---|---|---|
| 65536 | 3788 tok/s | 4775 tok/s | +26% |
| 131072 | 3788 tok/s | 4479 tok/s | +18% |
| 200704 | 3571 tok/s | 4108 tok/s | +15% |
| 253952 | 3441 tok/s | 3839 tok/s | +12% |

也就是說開快取的冷啟代價大約是15%，不是三成。

多輪Agent場景，每個會話5輪、200 token回覆（量於F=512，F=768下第一輪會再快約12%，後續輪次不受影響）：

| 上下文 | 第1輪 | 第2-5輪 | 會話總TTFT（快取／無快取） |
|---|---|---|---|
| 8192 | 2.50s | 0.21-0.37s | 3.6s／11.7s |
| 32768 | 6.55s | 0.23-0.25s | 7.5s／44.2s |
| 65536 | 17.03s | 0.30-0.46s | 18.6s／87.7s |
| 131072 | 34.11s | 0.36-0.42s | 35.7s／179.8s |
| 200704 | 55.45s | 0.59-0.85s | 58.4s／287.6s |

命中長度是精確的：每一輪回報的cached_tokens都等於`floor(前一次請求總長／1024)×1024`，40次量測沒有偏差。1024是dcp=4之下的排程區塊大小，尾端不足一塊的部分重算，多輪對話下幾乎無感。

### 檢索正確性

needle-in-a-haystack，5種上下文×8個深度共40格：

| | 準確率 |
|---|---|
| 無快取對照 | 39/40 |
| 開快取・冷啟 | 38/40 |
| 開快取・命中 | 38/40，與冷啟逐格相同 |

逐格相同是關鍵：不只總分一致，是哪兩格失敗、失敗成什麼樣都一致，因此快取沒有改變模型行為。兩邊共同的miss落在253952、深度0.2，那是256K邊緣的真實檢索極限，與快取無關。

## 兩組配置怎麼選

壓縮器狀態有兩種擺放方式，互斥：

| | PROFILE-P8 | PROFILE-CACHE |
|---|---|---|
| 壓縮器狀態 | 環形緩衝（位置取模） | 絕對位置 |
| prefix caching | 不可用 | 可用 |
| max-num-batched-tokens | 1024 | 768 |
| num-gpu-blocks-override | 650 | 1000 |
| long-prefill-token-threshold | 512 | 384 |
| 冷啟prefill | ~5200 tok/s | ~4100-4800 tok/s |
| 適用 | 批次、單次長文件 | Agent、多輪對話、RAG |

F在PROFILE-CACHE下的上限是768，這是實測出來的邊界，不是保守取值。引擎自己報的入場需求是：單條262144請求在F=512要0.34 GiB、F=768約0.45 GiB、F=1024要0.55 GiB；而物理可用池會隨F上升而縮小（activation變大），F=1024時只剩約0.89 GiB。同時204800的暖機需要約0.6 GiB的暫態餘裕。三個數字擺在一起，F=1024無解（0.55+0.6=1.15 > 0.89），F=768剛好過關。

順帶一提，`--gpu-memory-utilization`在這裡幫不上忙。只要`--num-gpu-blocks-override`設在計算值以下，池子大小就由override決定，util只影響那個被覆蓋掉的數字，實際VRAM用量裡沒有它。真正的槓桿只有三個：縮小池子（會撞入場門檻）、砍暫態（delta gather預算、logits上限、capture sizes、max-num-seqs）、降max-model-len。

互斥的原因是環形擺法以「絕對位置對視窗取模」定址，這種佈局與token前綴沒有對應關係，快取命中拿回的區塊放進環裡沒有意義。而環形擺法正是把每請求預留從858壓到262的關鍵，也就是F能開到1024的原因。要快取就必須放棄環形，F的上限也隨之從1024降到768。

對Agent而言這個交換划算。一個5輪、200k的會話，無快取時每輪都付57秒、總計287秒；開快取後只有第一輪付費（因F降級多付約11秒），後四輪合計不到3秒。單一會話省下150秒以上，會話越長差距越大。

如果工作負載是一次性的長文件處理，用PROFILE-P8即可，快取派不上用場，F=1024的吞吐才是重點。

## 環境

- 8×RTX 3090（24GB，SM86），無NVLink
- 兩顆PCIe switch串接，每顆四張卡。TP群組必須落在同一顆switch內
- Ryzen 5700X / 62GB RAM / Ubuntu 24.04 / driver 590.48
- PyTorch 2.13.0+cu130
- 模型：DeepSeek-V4-Flash-0731，fp8，約156GB

5700X的16執行緒餵八個worker偏緊。eager模式下可以觀察到明顯的排程straggler（單卡佔用率隨機掉底），開啟CUDA graph後大幅改善，但CPU更弱的平台可能會在這裡形成瓶頸。

供電與散熱是實際限制而非理論問題。八張卡在家用機殼內持續滿載時，會先撞到的牆是供電裕度和散熱，不是算力——這一點在規劃階段很容易被低估。實務上兩件事有效：壓測分輪跑（例如5分鐘一輪、輪間等GPU溫度回到基線再繼續），以及把功耗上限降下來（見下面的功耗調校，200W下八卡減載1200W，而prefill只掉不到兩成）。

## 功耗調校：200W是甜點

八張3090的原廠功耗上限是350W，滿載2800W。這台機器實際會先撞到的是供電而不是算力，所以值得認真調。我把350W到200W之間掃了一遍，用同樣的7個長度做等量比較：

| PL | 峰值prefill | 7點平均 | vs 350W | 邊際斜率 | 每瓦吞吐 | decode |
|---|---|---|---|---|---|---|
| 350W（原廠） | 5,023 | 4,617 | — | — | 1.65 | 50.09 |
| 280W | 4,812 | 4,423 | -4.2% | 0.21 %/% | 1.98 | 49.85 |
| 250W | 4,645 | 4,278 | -7.3% | 0.31 %/% | 2.14 | 49.87 |
| 220W | 4,383 | 4,036 | -12.6% | 0.47 %/% | 2.29 | 49.93 |
| **200W** | **4,129** | **3,800** | **-17.7%** | **0.64 %/%** | **2.38** | **49.88** |

「邊際斜率」是每降1%功耗要付出多少%吞吐。它從0.21一路陡到0.64，200W是拐點：再往下一階要付6.5-7%換10%功耗，而每瓦吞吐的增益已經從每階+0.33收斂到+0.08。性能的拐點和效率的拐點落在同一個地方。

**decode從頭到尾沒有動**（50.09→49.88，五個功耗點全在49.8-50.1之間）。這不是巧合：decode在batch 1下是記憶體頻寬與延遲受限，GDDR6X的頻寬不隨核心功耗縮。所以整條曲線上我們一直只在削核心頻率，沒有碰到使用者體感最敏感的那一項。這也給了一個很好用的停止訊號——**如果哪一階開始看到decode下降，那就是記憶體側開始被限制，不管prefill還剩多少都不該再降**。

代價換算成實際使用：一個200k的會話，第一輪prefill從約49秒變成約59秒；第二輪起走快取，命中TTFT的影響在0.1秒等級（命中只重算尾端1024個token，其中大部分時間是固定開銷）。換來的是八卡減載1,200W。對以快取命中為主的Agent流量，這個交換幾乎是免費的。

功耗上限重開機不會保留，而這台機器正好是「過熱跳電→冷啟」的模式，跳一次就回到350W、下次壓測又跳，會變成迴圈。所以要固化：

```bash
# /etc/default/nvidia-powerlimit
NVIDIA_POWER_LIMIT_W=200
```

```ini
# /etc/systemd/system/nvidia-powerlimit.service
[Unit]
Description=Apply NVIDIA persistence mode and per-card power limit
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/nvidia-powerlimit.sh

[Install]
WantedBy=multi-user.target
```

`nvidia-powerlimit.sh`做三件事：輪詢等`nvidia-smi -L`就緒（開機競爭條件）、開persistence mode（否則最後一個CUDA context結束時驅動卸載、限制被還原）、套用功耗上限。等待迴圈必須放在腳本裡而不是unit的`ExecStartPre`——systemd不解析`$(...)`，寫在unit裡整個unit會被拒絕，而錯誤訊息只會說「bad unit file setting」，不會告訴你是哪一行。

要跑極限數據時手動解開`sudo nvidia-smi -pl 350`，重開機自動回到200W。


## 24GB卡的記憶體陷阱

以下每一項都是實測得出，有幾項花了不只一次重開機才確認。

**VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0**。估算值保留1.46GiB，實測只用0.07GiB。單一收益最大的一項。

**VLLM_SPARSE_INDEXER_MAX_LOGITS_MB**。預設512，降到48-64無副作用。

**--num-gpu-blocks-override與錯誤訊息裡的available不是同一件事**。入場檢查回報的available是套用override之後的容量，物理容量要看日誌中的`Available KV cache memory`那一行。我在這裡誤判過兩次，以為是記憶體不足，實際上是自己把池子設得太小。

**PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True必須單獨使用**。與max_split_size_mb併用時expandable會被靜默停用，不會有任何警告。

**每次重開機都有暖機成本**。Triton JIT對每個新的chunk數量分桶都要編譯一次，每個新長度約7-11.5秒。部署SOP是開機後先跑暖機腳本（5個代表長度各發一次）再掛流量；自行量測時掃兩輪取第二輪。第一輪的數字會低一半，那不是真實性能。

## 已知行為

### 不保證跨次可重現

同樣的輸入、溫度0、序列提交，兩次執行可能產生不同的輸出。這與prefix caching無關——關閉快取時同樣發生，而且分歧得更早。

根因在上游vLLM。DSV4的稀疏索引器分數是`sum_h w_h · ReLU(q_h · k)`，任何不匹配任何注意力頭的條目分數恰好等於0.0。top-k的k是512，當正分條目不足512時，選擇的尾端是從一個巨大的「恰好0.0」平手池中取出；而上游的top-k核心以atomicAdd發配輸出槽位，哪個平手條目勝出取決於原子操作的到達順序。

實測（關閉快取、同prompt串行兩次）：首次分歧出現在生成的第9／0／2／1個token，分別對應8192／32768／131072／200704的prompt長度。

對品質的影響預期很小但不是零。被換掉的是索引器自身評為完全不相關（分數0）的條目，但它們仍佔用512個槽位之一，其KV仍進入注意力加總。實測needle準確率不受影響。

對可重現性的影響則是確定的。greedy解碼會把一次near-tie翻轉放大成完全不同的後續文字。做eval、回歸測試、A/B比對時必須知道這件事，否則會把噪聲當成訊號。

`VLLM_SM86_DET_TOPK=1`提供確定性的替代選擇器（穩定排序，平手取較低索引），但速度很慢，不建議在生產環境開啟。它也不是「正確答案」：低索引優先是另一種同樣任意的平手拆解，並帶有偏向序列前段的系統性偏差。它的正當用途是把這個變數固定住讓A/B比對有意義，本專案就是用它驗證prefix caching的正確性。

### 快取命中與冷啟不是位元等價的

即使把上面那個平手競態關掉，同一個prompt的快取命中結果與完整重算仍然不完全相同：token序列在前幾個token之後開始分歧，logprob幾乎每個位置都有微小差異（0.02-0.2量級）。

我們把嫌疑一個一個排除掉了：關掉上游的top-k平手競態（殘差不變）、修掉一個會越界寫入的KV區塊歸零缺陷（不變）、修掉delta-gather的allocator不對稱（不變）、關掉整個delta-gather路徑（不變）。剩下的唯一解釋是它本來就是「重用快取前綴」與「從頭逐塊建立前綴」之間的固有數值差異——兩種情況下注意力與壓縮kernel的分塊與規約配置不同，浮點結果就不同。這不是本分支特有的，是prefix caching這件事的性質。

影響範圍量過了：**不改變檢索正確性**（needle 40格網的cold與warm逐格相同，包含失敗的那兩格），**不改變快取的功能正確性**（命中長度40次量測零偏差）。它影響的只有「同輸入是否保證同輸出」，而那件事在這個堆疊上因為前一節的平手競態本來就不成立。

會提這件事，是因為它花了我們相當多時間去追，而追的過程順帶挖出三個真的缺陷（見下節）。如果你在自己的環境看到同樣的現象，不用再追一次。

### 併發：262144下只能單流，這是結構性的

開快取之後，262144的配置只能單流服務。原因不是「記憶體不夠用」，而是入場檢查的預留方式。

實測的行為是：**每條請求在10-20k以內時併發是真的**，超過就退化成錯開。分界點在10752（兩條交錯進行、聚合decode 66 tok/s）和20992（錯開、23 tok/s）之間。

機制是排程器admit新請求時要先預留滑動視窗群組的區塊，而**這個預留隨請求長度成長**。長度小的時候兩條都進得去，它們的prefill在同一步裡交錯、同時結束，然後一起decode；長度大的時候第二條要等第一條prefill完才進得來，於是第一條開始decode時第二條正在prefill，而一個被prefill佔滿的排程步驟只讓decode前進一個token——實測4.8 tok/s，對照單流的50。

診斷過程中最有價值的一個觀察：第二條被拒時`vllm:kv_cache_usage_perc`只有0.49。**usage算的是已寫入的區塊，入場檢查算的是已預留的**，兩者不是同一件事，看usage會以為還有一半空間。同樣的道理，我們一度以為加大池子就能解決，但override開到1600時NCCL在第一次使用通訊器時配置緩衝失敗（`ncclUnhandledCudaError`出現在PP的broadcast，不是乾淨的torch OOM，所以grep OutOfMemoryError是零）。1200確定安全，天花板在兩者之間，而那不足以讓兩條長請求同時入場。

沒開快取的PROFILE-P8可以併發2，是因為環形擺法把壓縮器狀態的預留壓成一個常數（約3個block/請求）而不是隨長度成長。所以這是ring與併發之間的結構交換。

三個相關的事實，一併記在這裡免得踩坑：

**聚合prefill永遠不會超過單流。** 每步只有一個token預算（F），N條併發是共享它而不是各拿一份。所以併發買到的是延遲的公平性，不是機器產出的增加。PROFILE-P8的併發2聚合prefill 4,909對照單流5,213，正是這個關係。

**`--long-prefill-token-threshold`必須小於等於F的一半**，否則一條prefill就吃光整個預算，第二條連在同一步裡排隊的機會都沒有。PROFILE-P8的threshold 512恰為F=1024的一半，這是它併發健康的必要條件之一。不過這只是必要不是充分：實測把threshold降到F/2（768→384）只讓10752那個點恢復（decode 35→66、prefill 3035→4157），更長的點沒有改變，因為那裡卡住的是預留而不是預算。降threshold對單流是免費的，`VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1`只在有兩條以上prefill排隊時才套用上限，實測單流prefill 4,646 vs 4,631，差異在噪聲內。

**併發4在任何長度都不是真的併發。** 早期PROFILE-P8的併發4數據（聚合decode 24.7-43.2，明顯低於併發2的63.5-80）當時被當成正常的併發衰減，回頭看是同一個入場瓶頸，只是那時沒有追根因。

要同時拿到ring的併發和絕對擺放的快取，需要一個「邊界前導群組」的設計：保留ring，另外掛一個很小的、可被前綴快取定址的群組，只存壓縮器在每個對齊邊界前要回看的那4列。評估過是可行的（約7-9個日曆週，主要風險在於需要一個「群組對命中長度棄權」的機制，那在hybrid coordinator裡沒有先例），但不在這個版本裡。

### 其他限制

併發4在任何長度都不是真的併發：四條的最壞情況預留遠超池子上限。我們早期PROFILE-P8的併發4數據（聚合decode 24.7-43.2，明顯低於併發2的63.5-80）當時被當成正常的併發衰減，回頭看是同一個入場瓶頸，只是那時沒有追根因。

FULL_AND_PIECEWISE的cudagraph在256K會OOM，差約2MiB。131k可開，但實測無增益——它的主場是混合步（一條prefill一條decode的步），同步起流的benchmark量不出來。

投機解碼裝不下，draft模型每卡多要0.9GB，沒有空間。

## 追殘差時挖出來的三個缺陷

都不是prefix caching引入的，是既有的，只是開了快取之後才有機會被觀察到。

**KV區塊歸零把「區塊步長」當成「寫入長度」。** DeepseekV4的packed佈局讓所有群組的所有層共用一塊slab，每層是一個有自己byte offset的strided view。歸零kernel卻用`stride(block_dim)`同時當作步長和寫入長度，所以任何offset大於0的層，每次歸零都會多寫offset個byte進下一個區塊——那個區塊可能正被別的請求使用——而最後一個區塊更會寫出配置範圍之外。修法是把步長和payload拆成兩張表，payload用stride span計算（這樣K/V-first重新排布的佈局也會留在同一個區塊內），並加上init時的斷言讓幾何不合法時大聲失敗而不是靜默損毀。

**歸零的判別式是精確type()比對，漏掉滑動視窗家族。** SWA的KV視窗和fp32壓縮器狀態都不在那個tuple裡，所以它們的新區塊只有在剛好被別的群組的tensor別名到時才被歸零。改成`isinstance(AttentionSpec)`，兩側（scheduler與worker）一致。

**delta-gather的`empty_cache`觸發次數無上限。** 原本用「prefill集合的邊緣」判斷，所以一條請求被跳過一步再回來就會再沖一次配置器，而`empty_cache`是裝置同步呼叫。在3條以上併發prefill的情況下這會反覆發生。改成每請求一次性的latch。同一個檔案裡BLOCKED sentinel的新鮮度判斷也被prefix caching自己打敗了（快取命中讓首次sighting帶著非零的prefix），改用與追蹤路徑相同的等值契約。

三個都有回歸測試，而且測試是先驗證過「在修復前的代碼上會失敗」才收下的。

## 對上游的發現

開發過程中在上游vLLM找到兩個問題，本分支均已修復。

**管線平行的靜默失步**。非輸出rank的例外被吞掉，導致跳過isend，產生永久的+1偏移。不會崩潰，只會安靜地輸出錯誤內容，屬於最難排查的一類。修法是重新拋出例外，並在張量字典中加入PP step-id契約做硬性檢查。

**worker例外被吞**。`multiproc_executor`對非回報rank的例外靜默處理，因此記憶體不足這類錯誤的表現是「引擎卡住」而非明確報錯，需手動pkill才能恢復。

## License and Credits

This fork inherits vLLM's Apache 2.0 license. New files carry `SPDX-License-Identifier: Apache-2.0`.

The flash-mla patch branch is MIT (Copyright (c) 2025 DeepSeek), compatible with Apache 2.0.

- Built on [haosdent/vllm](https://github.com/haosdent/vllm) — DeepSeek-V4-Flash support for vLLM
- The compression-preserving DCP design derives from Lasimeri's context-parallelism work; attribution notes are in `sm86_dcp_layout.py`, `sparse_attn_indexer.py` and `dcp.py`
- flash-mla sm86 sparse kernel integration references AppMana's consumer-GPU fork of flash-mla
