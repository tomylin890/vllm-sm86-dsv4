## 安裝

### 開始前

硬體與驅動的部分見上面的「環境」一節，這裡只列會擋住安裝的幾項：

- 八張SM86的卡，且TP群組不跨PCIe switch。下面兩組設定的每一個數字都是為八卡24GB這個配置量出來的。
- driver 590.48，venv裡的torch是2.13.0+cu130。
- Python 3.12。`pyproject.toml`宣告的是`>=3.10,<3.15`，3.12是上游文件與這台機器實際用的版本。
- nvcc。只有flash-mla那一步需要它，而且它的CUDA版本要對得上venv裡torch回報的版本（`python -c "import torch; print(torch.version.cuda)"`，這台是13.0）。vLLM本體不編譯，見下一節。
- 磁碟：模型fp8約156GB（P0當時從HF拉下來記錄是167GB、約20分鐘），加上預編譯wheel與flash-mla的build產物，留200GB以上。
- 模型`deepseek-ai/DeepSeek-V4-Flash-0731`本體，用你慣用的方式拉到本機即可，下面的啟動指令吃的是本機路徑。

`PYTHONOPTIMIZE`不要設。`python -O`會把assert整個剝掉，而有幾處幾何檢查正是靠assert做fail-close的：KV區塊歸零的「步長vs寫入長度」契約（`vllm/v1/worker/utils.py`）、以及hybrid coordinator那兩條dcp相關的斷言（群組型別、block size整除，也就是P11那個B1/B2）。剝掉之後不合法的設定不會炸，會安靜地算錯或安靜地開起來。rack kit的`verify_p11_B.sh`直接拒絕在`PYTHONOPTIMIZE`有值時執行，就是這個原因。

### 裝這個分支

wheel的底座是上游main的`62195e9784ebec1ece42b88a861734e0702cc2d5`。這個clone的HEAD相對它改了63個檔案（`git diff --name-only 62195e978..HEAD`），全部是`.py`——`csrc/`、`CMakeLists.txt`、`cmake/`一行未動。這63個裡有32個來自haosdent的DSV4 SM8x支援（`f8ea5bb16`，也是純Python），其餘43個是本分支。所以整個C++/CUDA擴充可以直接沿用那顆wheel，不需要在這台機器上編譯vLLM本體——那會花掉一小時起跳。

換句話說，`VLLM_USE_PRECOMPILED`在這裡合法的前提是「沒有人動過編譯單元」，不是「差異很小」。rebase之後第一件事就是重跑那個`git diff`確認仍然沒有非`.py`檔案，否則抽出來的`.so`與樹裡的Python就不是同一份。

對應的wheel先抓下來放本機。`VLLM_PRECOMPILED_WHEEL_LOCATION`吃URL也吃本機路徑（`setup.py`會先`os.path.isfile`判斷），放本機的好處是重裝不必再下載一次，而且哪次裝了哪顆wheel是看得見的：

```bash
mkdir -p ~/dsv4-dcp/wheels
curl -L -o ~/dsv4-dcp/wheels/vllm-0.26.1rc1.dev227+g62195e978-cp38-abi3-manylinux_2_28_x86_64.whl \
  'https://wheels.vllm.ai/62195e9784ebec1ece42b88a861734e0702cc2d5/vllm-0.26.1rc1.dev227%2Bg62195e978-cp38-abi3-manylinux_2_28_x86_64.whl'
```

venv用uv建，不帶`--seed`，所以裡面沒有`pip`模組——之後所有安裝都要走`uv pip install --python <venv的python>`，`python -m pip`會直接`No module named pip`。（樹裡`flash_mla_prefill.py`/`flash_mla_decode.py`的build提示寫的是`python -m pip`，那是假設venv有seed過，照抄會失敗。）

```bash
uv venv --python 3.12 ~/dsv4-dcp/venv

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=~/dsv4-dcp/wheels/vllm-0.26.1rc1.dev227+g62195e978-cp38-abi3-manylinux_2_28_x86_64.whl
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.26.1rc1.dev227+g62195e978

uv pip install --python ~/dsv4-dcp/venv/bin/python -e ~/dsv4-dcp/vllm --torch-backend=auto
```

`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM`不是可選的。這個clone裡沒有上游的版本tag，只有本專案自己的階段tag（`P4`/`P6`/`P7`/`P8`，而且是輕量tag，`git describe`不加`--tags`會直接說「沒有附註標籤」）。`git describe --tags`給出的是`P8-12-gdc5487ef2`——不是PEP440，setuptools-scm推不出版本號。用帶dist名的那個變數（`vllm`正規化後就是`VLLM`）而不是無範圍的`SETUPTOOLS_SCM_PRETEND_VERSION`，是因為後者會套到同一個shell裡之後每一個用setuptools-scm的build上去。

值就取底座wheel的版本字串。裝完`uv pip show vllm`回報的會是`0.26.1rc1.dev227+g62195e978.precompiled`——那個`.precompiled`後綴是`setup.py`在走預編譯路徑時自己加的（`vllm.__version__`不帶它，那是`_version.py`在加後綴之前就寫好的）。看到它就表示`.so`確實來自wheel而不是本機編譯。

如果哪天把分支rebase到別的上游commit，wheel URL和這個版本字串要一起換：`.so`是從wheel裡抽出來的，跟樹裡的Python必須是同一個base。

### flash-mla sm86

`VLLM_DSV4_FLASH_DECODE=1`（兩組設定都開）要的是patched fork的`fwd_sparse_decode_mla_partial`op，上游的flash-mla沒有這個op。decode之所以能在全上下文平坦，就是這顆核心。沒裝的話開機時warmup catalog那一項就會炸，例外訊息裡直接附build指令。

```bash
cd ~/dsv4-dcp/flash-mla-int
git checkout dcp-sm86-patches
git submodule update --init --recursive        # csrc/cutlass

~/dsv4-dcp/venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
FLASH_MLA_CUDA_ARCHS=86 uv pip install --python ~/dsv4-dcp/venv/bin/python \
  -v --no-build-isolation .

~/dsv4-dcp/venv/bin/python - <<'PY'
import torch, flash_mla
print(hasattr(torch.ops.flash_mla, "fwd_sparse_decode_mla_partial"))
PY
```

TODO：`flash-mla-int`的公開來源URL。工程筆記裡寫的是`git fetch <clone-remote> dcp-sm86-patches`，沒有記下remote，我無法從樹裡驗證，先不寫一個可能錯的網址。

三件事值得說明。`FLASH_MLA_CUDA_ARCHS=86`把nvcc釘在`sm_86`，那個repo的預設是`80`，用預設編出來的東西在3090上跑不到原生路徑。`--no-build-isolation`是必要的：這個擴充要對著venv裡那顆torch的ABI編，隔離的build環境會自己抓一顆torch，編出來的`.so`載入時才會爆。torch版本的檢查刻意放在build前而不是boot時：這個擴充走torch-stable ABI、支援torch>=2.9，venv裡比這舊就該當場停下來，而不是等到第一次decode才發現。

`--no-build-isolation`要求target venv裡有`setuptools`和`wheel`。上一節的vLLM安裝不保證會帶進來，缺了就`uv pip install --python ~/dsv4-dcp/venv/bin/python setuptools wheel`補上。

編譯時間會比上游的flash-mla久：decode那個translation unit每個split kernel要實例化兩次（`kPartial`）、mma kernel四次（`kFusedCombine`×`kPartial`）。

### 啟動

兩組設定的差別只有五處：F、`--num-gpu-blocks-override`、`--long-prefill-token-threshold`、prefix caching開關、以及ring那兩個環境變數。其餘完全相同。為什麼是兩組而不是一個開關，見上面「兩組配置怎麼選」。

引擎會擋住錯誤的組合：ring開著又開prefix caching，`compressor.py`直接拒絕啟動並在訊息裡把兩組設定各要怎麼配寫清楚。這個衝突不會被靜默解決，因為任一種解法都會讓KV佔用差好幾GB。

共用的環境變數區塊：

```bash
export VLLM_SM86_DCP=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export VLLM_DSV4_WARMUP=1
export VLLM_DSV4_SM86_INDEXER_TILES=1
export VLLM_DSV4_DELTA_GATHER=1
export VLLM_DSV4_DELTA_GATHER_BUDGET_MB=192
export VLLM_DSV4_FLASH_PREFILL=0
export VLLM_DSV4_FLASH_DECODE=1
export VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

| 變數 | 為什麼 |
|---|---|
| `VLLM_SM86_DCP=1` | 本分支所有DCP改動的總閘門。不開就是原版行為，dcp>1會被上游`mla/indexer.py`的`NotImplementedError: DCP is not supported with sparse indexer KV compression`擋下。 |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | 估算保留1.46GiB、實測只用0.07GiB。單一收益最大的一項，理由見「24GB卡的記憶體陷阱」。 |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | 索引器logits暫存上限，預設512。降到64省下的全是暫態，實測無副作用。 |
| `VLLM_DSV4_WARMUP=1` | 預設就是on，寫出來是因為它是「不要在量測請求裡付Triton JIT」的那一項：在權重載入並綁定cache之後，用dummy batch把已解析的稀疏MLA prefill/decode核心家族各跑一遍。它刻意走真的生產路徑、真的cache——Triton的特化不只吃constexpr，也吃cache stride與metadata形狀的對齊性質，拿假buffer編出來的特化對不上，等於白編。 |
| `VLLM_DSV4_SM86_INDEXER_TILES=1` | fp8 MQA logits核心改用消費級分支的SM86 tile配置，取代A100的autotune掃描。只調tiling與pipelining，reduction的block尺寸不動所以累加鏈不變；順帶移掉那個2-config autotune benchmark，那是boot-to-boot不確定性的來源之一。只在compute capability 8.6生效。 |
| `VLLM_DSV4_DELTA_GATHER=1` | prefill每個chunk只gather自上一個chunk以來新完成的壓縮entry，而不是整個prefix重打包重all-gather。壓縮entry在區塊邊界寫一次之後不再變動，所以staging的位元組一直有效。 |
| `VLLM_DSV4_DELTA_GATHER_BUDGET_MB=192` | 上面那些staging buffer的每worker總預算。某個(request, layer)超出預算就不追蹤它、退回全量gather，是逐請求逐層的降級而不是整體開關。P7-NOTES的預算表算出512「單條256k剛好放得下」，但那是只看staging自己；PROFILE-CACHE下ring關掉、每請求的滑動視窗預留大得多，實測要降到192才能讓204800的暖機不OOM。這是用約1%的gather效率換暫態餘裕，見「疑難排解」的記憶體章節。 |
| `VLLM_DSV4_FLASH_PREFILL=0` | prefill留在Triton pipeline，不走flash-mla的融合op。 |
| `VLLM_DSV4_FLASH_DECODE=1` | decode走patched fork的partial op，回傳本rank的pre-sink輸出加自然對數LSE，交給`dcp.py`的merge合併（sink在全域max處只折一次）。上一節那個build就是為了它。 |
| `VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE=1` | `--long-prefill-token-threshold`只在有兩條以上prefill在排隊時才套用，單流保持完整的F大小chunk。對單流是免費的（實測4,646 vs 4,631，在噪聲內）。 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 單獨使用。與`max_split_size_mb`併用時expandable會被靜默停用，不會有任何警告。 |

除錯用的`VLLM_SM86_DET_TOPK=1`很慢，只在需要把top-k平手競態固定住做A/B時才開，理由見「不保證跨次可重現」。

#### PROFILE-CACHE

```bash
unset VLLM_DSV4_COMPRESSOR_WINDOWED VLLM_DSV4_COMPRESSOR_WINDOW

~/dsv4-dcp/venv/bin/vllm serve <模型路徑> \
  --served-model-name dsv4-flash-0731 --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --decode-context-parallel-size 4 \
  --dcp-comm-backend a2a \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 768 \
  --max-num-seqs 4 \
  --long-prefill-token-threshold 384 \
  --num-gpu-blocks-override 1000 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}'
```

ring的兩個變數要`unset`而不是設成`0`：如果上一輪跑的是PROFILE-P8，shell裡的export會沿用到這一輪。

#### PROFILE-P8

```bash
export VLLM_DSV4_COMPRESSOR_WINDOWED=1
export VLLM_DSV4_COMPRESSOR_WINDOW=512
```

serve指令與上面完全相同，除了這四項：

```
  --max-num-batched-tokens 1024
  --long-prefill-token-threshold 512
  --num-gpu-blocks-override 650
  --no-enable-prefix-caching
```

`VLLM_DSV4_COMPRESSOR_WINDOW`必須是128的正倍數且嚴格大於128（128是最大的壓縮器回看窗，也是block table的token對齊），引擎會檢查。記憶體與W成線性，sub-chunk次數約`ceil(F/(W-128))`；W>=F可以讓壓縮器forward每層維持單一launch pair。512是實測用的值。

#### 每一條旗標在做什麼

| 旗標 | 為什麼 |
|---|---|
| `--served-model-name dsv4-flash-0731` | API裡看到的模型名，與磁碟路徑脫鉤，換路徑不必改客戶端。 |
| `--trust-remote-code` | DSV4的config與tokenizer帶remote code。 |
| `--kv-cache-dtype fp8` | KV存fp8。156GB權重之外每張卡剩不到1GB，bf16的KV放不下262K。 |
| `--block-size 256` | KV區塊的token數。這個值往上決定scheduler的區塊粒度，也決定prefix caching的命中長度是`floor(前次長度/1024)×1024`（1024=block_size×dcp）。 |
| `--host 0.0.0.0 --port 8000` | 綁在LAN上；只綁localhost的話從別台機器打會看起來「服務正常但連不上」。 |
| `--tensor-parallel-size 4` | TP群組落在單一PCIe switch內。TP8會在52-62k上下文處撞牆，原因就是TP的集合通訊跨了switch。 |
| `--pipeline-parallel-size 2` | 跨switch的那一刀放在PP，PP的通訊量比TP小一個數量級。 |
| `--decode-context-parallel-size 4` | 壓縮KV沿sequence維度round-robin分到4個rank；滑動視窗群組不分片、每rank複製。 |
| `--dcp-comm-backend a2a` | 預設的`ag_rs`是每層3次NCCL；`a2a`交換partial輸出與LSE後用Triton合併，降到2次。只對MLA模型有意義，且要求dcp>1。 |
| `--no-enable-flashinfer-autotune` | 關掉kernel warmup階段跑的FlashInfer autotune。這台上是純開機成本。 |
| `--tokenizer-mode deepseek_v4` | DSV4自己的tokenizer。 |
| `--tool-call-parser deepseek_v4` `--enable-auto-tool-choice` `--reasoning-parser deepseek_v4` | tool call與reasoning欄位的解析器。不跑Agent流量的話這三條可以拿掉，對效能無影響。 |
| `--max-model-len 262144` | 上下文上限。這個值直接進入記憶體帳，往下調是三個真正槓桿之一。 |
| `--max-num-batched-tokens` | 每個排程步的token預算，也就是全文說的F。prefill的chunk大小由它決定，是吞吐的主要旋鈕，同時也是記憶體帳裡activation那一項的驅動。 |
| `--max-num-seqs 4` | 同時可排程的序列數上限。262144下長請求只能單流（見「併發」一節），4留的是10-20k以內那段真併發的空間；它同時也是上面capture sizes只取到4的理由。 |
| `--long-prefill-token-threshold` | 單條prefill每步能吃的token上限。必須<=F的一半，否則一條prefill吃光整個預算，第二條連在同一步裡排隊的機會都沒有。 |
| `--num-gpu-blocks-override` | 直接指定KV池的區塊數，蓋掉profiler算出來的值。這是能不能開機的關鍵：太大會在NCCL配置通訊器緩衝時失敗（不是乾淨的OOM），太小會撞到入場門檻。 |
| `--gpu-memory-utilization 0.92` | 只影響那個被override蓋掉的數字，實際VRAM用量裡沒有它。保留是為了讓profiler的日誌數字有意義。 |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | 兩組設定的核心差異。這個模型在本分支上預設就是開的，兩邊都顯式寫出來，這樣開機指令本身是自我描述的，將來預設值翻轉也不會靜默改變設定。 |
| `--compilation-config` | `FULL_DECODE_ONLY`只對decode批次capture CUDA graph。`FULL_AND_PIECEWISE`在256K會OOM，差約2MiB。capture sizes只取[1,2,4]是因為`--max-num-seqs 4`已經把批次大小封頂在4，再往上capture只是白吃VRAM。 |

### 開機之後要暖機

開完機不能直接掛流量，也不能直接量測。Triton JIT對每個新的chunk數量分桶都要編譯一次，每個新長度約7-11.5秒，這筆錢是每次重開機都要重付的。引擎內建的warmup catalog（`VLLM_DSV4_WARMUP=1`）處理掉kernel家族本身，但走HTTP的那幾個形狀分桶還是要真的發請求才會碰到。

SOP是開機後對五個代表長度各發一次：

```
2048  16384  65536  131072  204800
```

`p11-rack-kit/verify_p11_A.sh`與`verify_p11_B.sh`的`warmup_sweep()`就是在做這件事（`P11_WARMUP_LENGTHS`），而且預設`P11_WARMUP_STRICT=1`——任何一發失敗就退出並拒絕往下跑，因為那一輪boot量出來的數字不可比。自己手動量測時掃兩輪取第二輪，第一輪的數字會低一半，那不是真實性能。

PROFILE-CACHE下多一件事：暖機本身會把cache填起來。之後任何需要冷啟數字的量測都必須用全新的prompt內容（rack kit的T系列腳本會給prompt加鹽）或重開一次機，否則量到的是命中。

啟動完成的判準是兩件事同時成立：日誌出現`Route: /v1/models`，且`GET /v1/models`真的回200。只看其中一個都會誤判——路由掛上了不代表引擎已經能接請求，而在路由掛上之前連線被拒也不代表啟動失敗。

### 用rack kit啟動

`p11-rack-kit/`裡的兩支腳本可以直接拿來開機，它們額外做了pkill、等VRAM排空、READY輪詢、暖機掃描、以及開機事實檢查（群組數、scheduler block size、`num_gpu_blocks`），失敗會給出退出碼而不是留一個半死的引擎。

要注意腳本的預設值是P11驗證期的sizing（B是F=512、override 720），不是最終production的PROFILE-CACHE。要跑上面那組數字得顯式覆寫：

```bash
P11_F=768 P11_OVERRIDE=1000 P11_LONG_PREFILL_THRESHOLD=384 \
  bash p11-rack-kit/verify_p11_B.sh
```

這兩支腳本還會從`~/dsv4-dcp/verify_mc5.sh`讀模型路徑與delta-gather預算（`P11_REF_SCRIPT`），那個檔案只在機器上、不在repo裡。沒有它就用`P11_MODEL=`直接指定，預算會退回512並印一行warn。
