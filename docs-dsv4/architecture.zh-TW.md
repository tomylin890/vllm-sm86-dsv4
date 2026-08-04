## DeepSeek-V4-Flash的KV長什麼樣

要理解這個分支做了什麼，得先知道這個模型的KV跟一般的注意力有多不一樣。以下每一項都以代碼為準（`vllm/models/deepseek_v4/`），論文與代碼衝突處一律取代碼，衝突清單在`ARCHITECTURE.md`第9節。

### 名字叫MLA，實際上是shared-KV MQA

`num_key_value_heads=1`，一個512維的KV entry同時當K和V用，config裡沒有`kv_lora_rank`也沒有`v_head_dim`。vLLM把它標成`is_mla()=True`只是為了複用MLA的cache管線，不要把V3那套MLA的假設帶進來。

head_dim 512 = 448維NoPE + 64維RoPE，query head有64個，softmax scale是1/√512。entry在cache裡佔584 byte：448維FP8（量化block 64，所以是7個真的UE8M0 scale加1個pad byte）加64維BF16的RoPE，資料段對齊576。`compressor.py`的`_token_stride`（448+64×2=576）和`_scale_dim`（448//64+1=8）就是這個算式。

### 兩種壓縮比，兩種完全不同的性質

主幹43層。L0與L1是純滑動視窗（`compress_ratio`為0，代碼取`max(1, ...)`當1）；L2到L42在CSA(m=4)與HCA(m'=128)之間交替，21層C4、20層C128。另外3層DSpark draft也是純滑動視窗。

C4是**重疊式**壓縮：每個entry由2m=8個token加權而成，跨進前一個block，權重來自單一的聯合per-dimension softmax（512個維度各自一組凸組合）。C128不重疊。代碼裡就是`self.overlap = compress_ratio == 4`與`self.coff = 1 + self.overlap`，`coff`從此貫穿所有的維度與視窗計算。

其餘幾條規則在分片時全都會變成約束：

- block完成條件是`(pos + 1) % m == 0`，entry的RoPE位置是`m·i`（block首token的絕對位置）。
- 位置偏置`ape`以**絕對位置mod m**索引。任何重新分片都必須保持position-mod-4對齊。
- 因果可見性是`s < (t+1)//m`——query看得到自己剛剛完成的那個block。論文寫的是`Floor(t/m)`，代碼不是。

### 稀疏索引器

只有C4層有索引器（`attention.py`：`if self.compress_ratio == 4`）。它有自己的一整套壓縮器：head_dim 128、同樣的m=4、自己的W與`ape`、Hadamard rotation後量化。cache是FP8的132 byte/entry（128 byte加4 byte的fp32 scale）；還有一個68 byte的MXFP4佈局，但SM8x不准用，DCP寫入路徑對它直接`NotImplementedError`。

分數是`Σ_h w_h · ReLU(q_h · k)`，**沒有softmax**，64個head。top-k是全域的，k取`config.index_topk`（Flash是512）。SM8x走Triton fallback（`fp8_mqa_logits_triton`/`fp8_paged_mqa_logits_triton`，fp32 logits）。

「沒有softmax」加上「ReLU」是README〈已知行為〉裡那個平手競態的來源：不匹配任何head的條目分數恰好是0.0。

### 滑動視窗群組

視窗128，block 64。block size不是自由參數——SWA與C4A的KV共用同一塊物理tensor，C4A的block形狀是`[256//4, head_dim] = [64, head_dim]`，所以SWA也必須是64。

它**與壓縮分支共用同一個softmax**：indices串接、單次sparse attention呼叫。DCP下這件事的意思是SWA的貢獻必須當成merge裡的一個(m, l, o) partial，絕對不能單獨歸一化。

Attention sink是每個head一個fp32的可學logit，**只加進softmax分母、只加一次、以全域running max為基準**。

### fp32的壓縮器狀態

壓縮計算全程fp32：W^KV、W^Z、`ape`、state buffer、softmax、加權和，之後才RMSNorm→RoPE（後64維）→FP8量化寫入。bf16化壓縮等於每個entry都偏。

關鍵是這個state**本身是一個paged KV群組**，spec是`SlidingWindowMLASpec`，dtype有fp32硬斷言（`CompressorStateCache`）。它的幾何也被page共享綁住：

| 狀態群組 | state_dim | block_size | sliding_window | 每token |
|---|---|---|---|---|
| C4壓縮器（注意力） | 2·coff·512 = 2048 float | 4 | 8 | 8192 B |
| C128壓縮器 | 2·1·512 = 1024 float | 8 | 128 | 4096 B |
| C4壓縮器（索引器） | 2·coff·128 = 512 float | 4 | 8 | 2048 B |

壓縮kernel在每個邊界位置`p`會gather`[p − (1 + OVERLAP)·m + 1, p]`這`L = sliding_window`列，其中`L − m`列落在`p`之下。這一個事實同時驅動了P8和P11，值得先記住。

### 加起來的群組表

| 群組 | spec | block | 內容 | DCP |
|---|---|---|---|---|
| C4A壓縮KV | `MLAAttentionSpec` cr=4 | 256 token / 64 entry | 584 B/entry uint8 | 分片 |
| C128A壓縮KV | `MLAAttentionSpec` cr=128 | 256 token / 2 entry | 同上 | 分片 |
| 索引器KV | `MLAAttentionSpec` cr=4 | 256 | 132 B/entry | 分片 |
| SWA KV | `SlidingWindowMLASpec` window 128 | 64 token | 原始KV，與C4A共用物理page | 複製 |
| C4壓縮器狀態 | `SlidingWindowMLASpec` fp32 window 8 | 4 token | 8192 B/token（索引器版2048） | 複製 |
| C128壓縮器狀態 | `SlidingWindowMLASpec` fp32 window 128 | 8 token | 4096 B/token | 複製 |

排程器端會把幾何相同的群組合併，所以實際的scheduler group數比表列少（索引器的C4狀態與注意力的C4狀態幾何相同）。真正要記的是那組block size：**256、64、4、8**。它決定了`scheduler_block_size = lcm(各群組block × dcp)`——dcp=4時是1024（README裡那個「命中長度對齊1024」就是它），dcp=8時是2048——也決定了開快取之後`hash_block_size = gcd(...)`只能是4。

## 為什麼vLLM的context parallelism接不上

上游的DCP假設一個同質的full-attention KV群組沿序列維度分片。這裡有六種幾何、三種dtype的混合群組，而且其中只有三組**可以**分片。具體卡在四個地方：

**一、群組型別。** `HybridKVCacheCoordinator`在`dcp_world_size > 1`時斷言每個群組是`FullAttentionSpec`或`MambaSpec`。`MLAAttentionSpec`繼承`FullAttentionSpec`所以三個壓縮KV群組過得去；`SlidingWindowSpec`直接繼承`AttentionSpec`，所以SWA和兩個狀態群組全部撞牆。這是「還沒有DCP-aware處理」的一刀切守衛，不是針對這個模型。

**二、壓縮器狀態不能分片。** 它不是KV，是一個以絕對位置定址、每個邊界要回看`L`列的fp32累加器。round-robin分片會把回看窗切碎。SWA同理——每個rank都必須看得到完整的128個token，因為它與壓縮分支共用softmax。所以需要一個「這個群組複製、不分片」的概念，這就是`dcp_exempt`。

**三、SM8x的decode kernel不輸出LSE。** `rocm_sparse_attn_decode`把SWA當main段、壓縮top-k當extra段，跑單次online softmax，sink在kernel內套用，只寫normalized bf16——`m_i`/`l_i`留在暫存器裡就丟了。沒有(m, l)就沒有辦法跨rank合併。

**四、索引器的top-k是全域的。** 每個rank對自己的shard取top-512，聯集不等於全域top-512。上游對這件事有明文拒絕：`vllm/v1/attention/backends/mla/indexer.py`的`NotImplementedError: DCP is not supported with sparse indexer KV compression`。

再疊一層精度約束，這些不是「接得上」的問題而是「接上之後對不對」的問題：sink只能在全域merge處、以全域max為基準、加一次；merge必須fp32且rank順序固定（DeepSeek為了batch invariance明言禁用split-KV attention，決定性得我們自己保住）；全域絕對位置必須貫穿`ape`的mod-4、entry的RoPE位置和YaRN表，per-shard從0重新編號是靜默的精度bug；KV bytes是函數的一部分，量化一次、以量化形式搬運，絕不用不同的grouping重新量化（本層quant block是64，DeepGEMM預設是128）。整理成12條在`ARCHITECTURE.md`第10節，實作時當checklist用。

## 分片的形狀

分片發生在**壓縮entry空間**，不是token空間。唯一來源是`vllm/v1/attention/backends/mla/sm86_dcp_layout.py`：

```
owner(e)       = (e // I) % W
local_entry(e) = (e // (I·W))·I + e % I
global(r, j)   = (j // I)·(I·W) + r·I + j % I
```

`W`是dcp world size，`I`是`cp_kv_cache_interleave_size`，全程鎖1（上游警告interleave>1連V3.2都過不了gsm8k parity，不要碰）。`-1`是無效哨兵，穿過每個helper不變。

順序很重要：**先除以m，再localize**。反過來會少算——world=2、rank 0、m=4、全長12時，正確答案是2個entry（{0, 2}），但`localize(12)//4`只給1。這是P2b修掉的。

shard邊界必須落在128的倍數上，`scheduler_block_size`（dcp=4是1024）自動滿足。

## 各階段做了什麼

順序就是實際的開發順序，每一步都是前一步的閘門通過之後才開始的。P1到P9每一節對應一份`P*-NOTES.md`，P11與P12對應的是`-DESIGN.md`，P10只有commit訊息。中間跳過的兩個編號：P3是`PLAN.md`裡的容量驗證閘門，只留下一個把short-context top-k快速路徑接回DCP producer的修正commit，沒有獨立文件；P5評估過MoE的Marlin INT8啟動路徑並且**否決**，結論記在`P5-NOTES.md`，零程式碼改動。

### P1 — KV群組管路與dcp_exempt

問題：引擎起不來。

做法是一個判別式：`is_dcp_exempt_spec`（`vllm/v1/core/kv_cache_utils.py`）。所有必須對「這個群組是分片還是複製」有共識的地方都呼叫它——manager的真實block size、排程器端的群組block size、worker端的block table列寬——所以它們不可能漂移。周邊的管路：

- `vllm/v1/kv_cache_interface.py`：`SlidingWindowSpec.max_memory_usage_bytes`那個`dcp == 1`的斷言在gate下放行（未分片的per-rank大小正好就是複製的footprint）；新增一個gated的`max_num_blocks_per_req`回傳未分片的列寬。
- `vllm/v1/worker/block_table.py`：`BlockTable`多一個`shard_dcp`參數（強制world=1/rank=0，於是`compute_slot_mapping`把每個token都存在本地），`MultiGroupBlockTable`多一個`dcp_exempt`列表。
- 經`gpu_input_batch.py`串到`gpu_model_runner.may_reinitialize_input_batch`，那裡逐群組算出exempt並納入reinit的觸發比較。

買到的是：dcp=2能init，而且KV pool顯示約2倍的token數、KB/token維持在壓縮級——後者是「靜默失去壓縮」的檢查，比前者重要。

這一步同時埋了一顆地雷：`resolve_kv_cache_block_sizes`對每個`AttentionSpec`乘dcp，包括我們讓manager跳過縮放的那些exempt群組。關快取時完全看不到，開了就在啟動時炸。見P11的B2。

### P2a — decode端的跨shard合併

問題：kernel不吐LSE。

`rocm_sparse_attn_decode`加一個`return_softmax_stats=True`（kernel內是`EMIT_SOFTMAX_STATS` constexpr），輸出pre-sink的fp32 `(m, l)`，形狀`[T, H]`。新檔`vllm/models/deepseek_v4/common/ops/dcp.py`把它轉成`lse = m + log(l)`，接上樹內既有的`cp_lse_ag_out_rs`/`dcp_a2a_lse_reduce`，最後`apply_attn_sink`用`logaddexp`在全域max上把sink折進去一次。

兩個容易寫錯的細節：空shard的哨兵是有限的`-1.0e30`而不是`-inf`（`-inf − -inf = NaN`會毒害整條softmax代數，而`exp(-1e30 − g)`在fp32下就是0，該drop的自然drop掉）；LSE的底數約定全程是自然對數（`is_lse_base_on_e=True`），弄錯會靜默毀掉結果而不是報錯。

Q要all-gather（`_maybe_gather_dcp_q`）：TP切query head，DCP在TP群組內切KV，所以每個rank的KV shard必須面對整個群組的所有head；a2a的LSE合併再把head輸出送回owner rank。

買到的是一個與「單一全域softmax加一次sink」數值等價的decode。純python模擬對照過四個case（含兩個空shard、全空加sink、sink接近max），最大絕對差2e-14。

### P2b — 全域top-k

`_sm86_dcp_global_topk`（`vllm/model_executor/layers/sparse_attn_indexer.py`）：各rank對自己的shard取local top-512，以固定rank順序all-gather fp32分數加int32全域entry id（固定寬度`topk × dcp`），再取全域top-512。

`torch.topk`不保證平手順序，所以換成兩次stable argsort——先升序全域index，再穩定降序分數——分數相同時取較小的全域entry index。無效候選用分數`-inf`加`int32`最大的排序index，這樣一個真的拿到`-inf`分數的候選仍然贏得過padding。

驗收標準是「選中的index集合完全相等」加greedy逐token相同，**不是logit MSE**。分數的浮點差在這裡沒有意義，選中誰才有。

同一階段還修了decode seq len的compress-then-localize順序，以及新增rank-local的壓縮entry slot mapping（`_sm86_dcp_compressed_slot_mapping`），並把索引器裡那個`compress_ratio > 1`的DCP守衛改成gate下放行。

### P2c — 寫入端

壓縮entry的寫入要照ownership過濾並改寫slot。compressor的forward在gate下強制走Triton launcher（cutedsl的SM90+路徑與ROCm的two-stage路徑沒有CP layout支援），kernel拿到`DCP_WORLD_SIZE`/`DCP_RANK`/`DCP_ENTRY_INTERLEAVE`三個constexpr，逐邊界token算出全域entry index`e = position // m`，不屬於自己就early return，屬於自己的就在kernel內從P1的分片block table算出rank-local slot。

每個rank都用fp32算出**每一個**entry，只存自己擁有的。計算是重複的，但不需要任何collective、也就capture-safe——這個取捨是刻意的。

### P2d — prefill的all-gather與C128A decode metadata

ownership代數在這一步從索引器裡抽出來，變成`sm86_dcp_layout.py`這個唯一來源（純搬移，無邏輯變更）。之後的prefill gather、C128A decode metadata、P4的虛擬block table、P7的delta planner都從那裡拿同一組公式。

### P2e — 確定性top-k（除錯用）

`VLLM_SM86_DET_TOPK`，選擇器在`vllm/v1/attention/ops/sm86_det_topk.py`。它為什麼存在、以及它為什麼不是「正確答案」，README的〈已知行為〉已經說了。這裡只補一句用途：驗證prefix caching正確性時，就是靠它把上游那個平手競態固定住，A/B才有意義。

### P2f — CUDA graph

P2a在`torch.cuda.is_current_stream_capturing()`下放了一個無條件的`RuntimeError`。P2f把它拆掉：逐op分類（in-place、persistent buffer、graph-private pool、host常數），把所有輸出接到capture之外配置的持久buffer。

過程中挖出真正的原因——`sparse_mla.py`裡C128A的decode寬度沒有釘死。那才是不能capture的東西，P2a的理由（「每步都是新的ragged tensor」）在這個base上其實不成立，因為非DCP的C4A decode本來就在capture裡跑。

買到的是DCP decode能跑`FULL_DECODE_ONLY` graph：eager下4-6 tok/s，開graph約70。

### P4 — prefill的固定開銷

診斷先於修法。DCP prefill比dcp=1慢約3.3倍，而且per-forward成本在5.6倍的prefix長度範圍內是**平的**（360-460 ms）。成本不隨長度變化就是每層固定開銷的簽名——不是頻寬，也不是kernel慢。一次2730 token、F=256的prefill量到約72,800次額外kernel launch、7,216次硬同步、913次NCCL collective。

四項優化：

1. 虛擬block table改成閉式，不再用16次`nonzero()`逐rank堆出來（那16次就是那16個硬同步）。
2. 把它memoize。它只依賴`(max_entries, max_local, num_reqs, world, interleave, device)`六個值，與層無關、與KV資料無關，卻在每次forward為全部41個壓縮層各重建一次。LRU上界同時卡條目數與元素數，因為chunked prefill會讓`max_entries`一路往上走、key是真的會churn的。
3. 索引器DCP合併在「全域壓縮entry總數不超過topk」時short-circuit——每個候選都會被選中，合併是identity，可以整個跳過（含兩次NCCL）。gate用的量是host的`max_seq_len // compress_ratio`，預設`-1`表示「沒提供」，所以未供給時short-circuit是關的而不是誤判成0。
4. `get_dcp_local_seq_lens`的`dcp_rank`不再包成device tensor。`torch.tensor(scalar, device=cuda)`會從pageable memory做一次阻塞H2D，而這條路徑每個壓縮層每個chunk都跑一次。

結果是cache hit的層（41層裡的39-40層）在這條路徑上是0個torch op、0次同步、0次H2D。

### P6與P9 — flash-mla的sm86核心

P6是prefill：一次`flash_mla.sparse_mla_prefill`（`torch.ops.flash_mla.fwd_sparse_prefill_mla`）取代原本「bf16 dequant workspace＋`combine_topk_swa_indices`＋`rocm_sparse_attn_prefill`」三件套，每chunk每層一次。SWA與壓縮兩條串流在kernel內共用單一softmax（第6節那條鐵律），sink在softmax初始化時折進去一次。adapter在`vllm/models/deepseek_v4/ampere/flash_mla_prefill.py`，旗標`VLLM_DSV4_FLASH_PREFILL`。

P9是decode：往flash-mla fork加一個**新的**op `fwd_sparse_decode_mla_partial`，輸出per-rank normalized的pre-sink `out`加自然對數的fp32 `lse`——正好是P2a那條merge路徑要的形狀。所以vLLM這側只是把`rocm_sparse_attn_decode + softmax_stats_to_lse`換成一次op呼叫，`dcp_merge_flashmla_output`與`self.attn_sink`完全沒動。原有的op沒碰。adapter在`ampere/flash_mla_decode.py`，旗標`VLLM_DSV4_FLASH_DECODE`。同一批patch還加了`BLOCK_M=16`的prefill實體化，在`num_heads <= 16`時派送。

README那個「decode從短上下文到254k幾乎不動」就是從這裡來的：成本只跟選中的512個entry有關，與序列總長度無關。

P6另外附了兩件周邊：`VLLM_DSV4_WARMUP`（預設開）把整個已解析的prefill/decode kernel家族在引擎init時跑一次，catalog在`vllm/model_executor/warmup/dsv4_sm86_warmup.py`；`VLLM_DSV4_SM86_INDEXER_TILES`則是因為上游那組Triton indexer autotune config是對A100/SM80調的。P9另外把prefill每個chunk數量分桶都要重編一次的成因（隨桶變動的constexpr）釘死。這幾項處理的是同一件事的不同層次，README〈記憶體陷阱〉那節講的部署暖機SOP是這些之後仍然存在的殘量。

flash-mla那側的改動在一個獨立的fork分支（`dcp-sm86-patches`）上，用`FLASH_MLA_CUDA_ARCHS=86`建進venv，build步驟見「安裝」一節（那裡也記著同一個待補的URL）。

### P7 — delta gather

問題：`ampere_sparse.py`把`gather_lens=None`傳進壓縮entry的gather，於是每個prefill chunk都重新pack、重新all-gather、重新消費**整個**prefix。壓縮entry在它的block邊界寫入一次之後永不變動，所以第一次之後的重傳全是冗餘，總量`O(P²/(m·F))`。

做法：每個(request, layer)一個持久的`[capacity, 584]` uint8 staging，第`e`列就是全域entry `e`的584個raw byte（576資料＋UE8M0 scale在+576，正好是一個`block_size = 1`的page）。列號跨chunk穩定，所以index轉換退化成identity——這是選全域entry順序而不是per-chunk gathered順序的理由，後者的`max_local`一長，所有id就都位移了。每個chunk只pack/all-gather/scatter新增的`[prev, new)`。bytes與scale全程原樣搬運，唯一的dequant還是原本那一次。

生命週期不靠scheduler hook（`vllm/models/deepseek_v4/ampere/dcp_delta_tracker.py`）：GC條件是「這一步的prefill列裡沒出現」——結束、abort、搶佔、轉進decode全都表現為缺席。另外用一個prefix連續性檢查讓過期狀態在結構上不可能存在：每次sighting的prefix token數必須等於上次sighting記下的seq len，chunked prefill保證這件事，任何違反就重設為全新admission。有per-worker位元組預算（`VLLM_DSV4_DELTA_GATHER_BUDGET_MB`），charge失敗就讓那個(request, layer)退回完整re-gather，是逐請求逐層的優雅降級而不是整體失敗。

README〈三個缺陷〉裡的第三個（`empty_cache`觸發次數無上限）就在這個檔案。

### P8 — 壓縮器狀態的環形緩衝

問題：預設擺法給每個被排程的token一列paged row，所以一步F個token會逼出`F + L − 1`列live row，per-request的預留隨F線性成長。

做法：一個W個token的環，`slot = position % W`，並把compressor forward的每一步切成`G = W − L + 1`個flat batch token的子chunk。`L`就是kernel的gather span（`coff · m`）。子chunk內每個寫都在每個讀之前，live set最多`(G − 1)`個新寫加`(L − 1)`個回看加1，也就是`G + L − 1`個相異位置，所以`W ≥ G + L − 1`時無碰撞。這個界是**緊的**——正好從邊界位置起頭的子chunk在`G = W − L + 2`就開始壞。

切在flat token軸上是per-request安全的關鍵：每個request的token在batch裡是一段連續遞增的run，所以G個token的flat slice對任一個request最多貢獻G個**連續**位置。

預留因此塌成常數`cdiv(W, block_size)`，也就是README裡那個858→262。

驗證全部是`vllm_config`的純函數，所以每個rank算出一樣的答案，排程器看到的admission數字保持rank-invariant。兩個拒絕都是硬raise而不是降級：

- 與prefix caching互斥（原因README已說：`pos % W`的slot本來就不是prefix-addressable）。降級成`None`會讓KV footprint差好幾GB，而且要很久之後才會以admission拒絕或OOM的形式浮出來。
- 要求hybrid KV cache manager必須開著。`_promote_local_kv_cache_specs`會把`SlidingWindowMLASpec`重建成`MLAAttentionSpec`而帶不動`state_window`，於是寫入端（slot mapping，由spec推導）回到絕對位置，讀取端（compressor，由env推導）繼續folding——靜默的數值錯誤。這個旗標也會被**隱式**設定（沒有HMA支援的KV connector、不支援hybrid的平台），不只是CLI。

### P10 — 自適應的long-prefill門檻

`--long-prefill-token-threshold`是全域生效的。單流時它把chunk數量加倍，200k的needle從44秒變成59秒（-34%）。但它同時又是併發健康的必要條件（README〈併發〉那節：threshold必須小於等於F的一半，否則一條prefill就吃光整個預算）。

`VLLM_LONG_PREFILL_THRESHOLD_ADAPTIVE`（預設off，等同上游語意）只在有兩條以上處於prefill階段的請求排隊時才套用上限，單條請求維持完整的F。pending數＝waiting queue加上running中`num_computed < num_prompt`的請求，每次`schedule()`算一次。改動在`vllm/v1/core/sched/scheduler.py`。

### P11 — prefix caching

這一階段的研究推翻了原本的設計，值得完整記下來，因為兩個否定都不直覺。

原案是「命中之後往回trim掉幾個block再重算」。

**在ring下trim不夠。** 壓縮kernel的gather是`start = position − (1 + OVERLAP)·m + 1`，唯一的守衛是`mask_pos = pos >= 0`。C4的m=4、L=8，任何m對齊的續算點R之後第一個邊界是`p = R + 3`，它的gather從`R − 4`開始。`[R−4, R−1]`這四列被讀但永遠不會被重算——不管trim 256、1024還是8192，缺口只是被搬到別的位置，這個赤字是尺度不變的。而且新配置的狀態block當時不在歸零名單裡（`SlidingWindowMLASpec`被刻意排除在那個`type()`比對之外），所以那四列是被當成fp32讀的uint8 KV byte，嚴重度是Inf/NaN級而不是「稍微偏一點」。

**在預設擺法下trim不必要。** `SlidingWindowManager._contiguous_blocks_for_hit = cdiv(window − 1, block_size)`在命中邊界H保留的正好就是回看窗：C4狀態`cdiv(7,4)=2`個block，也就是`[H−8, H)`，嚴格包含需要的`[H−4, H−1]`；C128狀態`cdiv(127,8)=16`個block即`[H−128, H)`，而C128沒有overlap，它第一個邊界`p = H+127`讀`[H, H+127]`，H以下什麼都不需要；SWA KV `cdiv(127,64)=2`個block就是整個128視窗。而H是`scheduler_block_size`的倍數。

所以prefix caching不是靠trim拿到的，是靠**放棄環形擺法**。這就是README〈兩組配置怎麼選〉那個互斥的來源。

真正的阻塞是兩個和壓縮器完全無關的啟動錯誤：

- **B1（上游）**：前面說過的`HybridKVCacheCoordinator`型別斷言。它今天看不到，只是因為關快取時`get_kv_cache_coordinator`路由到`KVCacheCoordinatorNoPrefixCache`。
- **B2（我們P1造成的）**：`resolve_kv_cache_block_sizes`對每個`AttentionSpec`乘dcp，包括那些exempt群組。開快取後函式不再early return，`hash_block_size = gcd(1024, 256, 16, 32) = 16`，而manager的block size是`[1024, 64, 4, 8]`，coordinator的整除斷言在`4 % 16`炸掉。P1當時留的註解說「較粗的LCM是安全的superset對齊，這裡不需要gate」——那對LCM成立，對GCD剛好是反方向。實測時B2還比B1先炸，因為整除斷言排在型別斷言前面。
- **B3（後果，不是bug）**：修好之後`hash_block_size = gcd(1024, 64, 4, 8) = 4`，被C4狀態群組那個受page共享約束的block_size 4釘死，沒有更粗的選項。262k的prompt因此要在排程器的關鍵路徑上算約65,536個鏈式block hash。

config時間的檢查放在`compressor.py`：`validate_compressor_lookback_coverage`把上面那組算術寫成會大聲失敗的斷言（命中預留必須涵蓋回看、`scheduler_block_size`必須整除每個視窗與每個m），`check_compressor_kv_cache_config`在啟動log印出profile名稱與三個block size，所以部署矩陣可以直接從log斷言。今天這些條件成立是1024/128/8/4的算術巧合——任何一個數字動了就該炸，那正是這兩個函式存在的理由。

### 沒做的：P12

README提過的「邊界前導群組」——保留ring，另外掛一個很小的、可被前綴快取定址的群組，只存壓縮器在每個對齊邊界前要回看的那幾列。完整設計（11個實作步驟、對抗性審查、工程量與風險）在`P12-DESIGN.md`。要接手「ring與快取兼得」的人請從那裡開始，不要從頭想。

## 從哪裡讀起

分支是`dcp-sm86`，基底是`f8ea5bb16`（haosdent的DSV4 SM8x支援）。`git diff f8ea5bb16 HEAD`就是這個分支的全部——43個檔案，全部是`.py`，其中11個是新增的（7個原始碼、4個測試；另有1個既有測試被改）。

| 檔案 | 負責什麼 |
|---|---|
| `vllm/v1/attention/backends/mla/sm86_dcp_layout.py` | ownership代數的唯一來源，先讀這個 |
| `vllm/models/deepseek_v4/common/ops/dcp.py` | 跨rank的LSE合併與sink |
| `vllm/models/deepseek_v4/ampere/ampere_sparse.py` | prefill/decode的DCP路徑總入口 |
| `vllm/models/deepseek_v4/compressor.py` | 壓縮器、狀態群組spec、P8的ring、P11的config時檢查 |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | 壓縮/量化/寫入的fused kernel，回看窗在這裡 |
| `vllm/models/deepseek_v4/ampere/dcp_delta_tracker.py` | P7的staging生命週期 |
| `vllm/models/deepseek_v4/ampere/flash_mla_{prefill,decode}.py` | flash-mla op的adapter |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 索引器的全域top-k合併 |
| `vllm/v1/attention/backends/mla/indexer.py` | 索引器metadata的DCP在地化 |
| `vllm/v1/core/kv_cache_utils.py` | `is_dcp_exempt_spec`、block size解析 |
| `vllm/v1/worker/block_table.py` | `shard_dcp`／`dcp_exempt`／ring的slot mapping |

文件對應：`PLAN.md`（階段切分與閘門）、`ARCHITECTURE.md`（模型與精度鐵律）、`PORT-NOTES.md`（P1與後續的修正紀錄）、`P2a`到`P2F`（decode側DCP）、`P4`（prefill開銷）、`P5`（被否決的Marlin INT8路徑）、`P6`／`P9`（flash-mla）、`P7`（delta gather）、`P8`（ring）、`P11-DESIGN.md`（prefix caching的研究與設計）、`P12-DESIGN.md`（未實作的前導群組）。

每一份notes都明確標示了哪些結論是靜態推導、哪些是在機器上實測的——這台開發機沒有CUDA，很多階段是先靜態完成再上機驗證的，讀的時候要注意這個區分。
