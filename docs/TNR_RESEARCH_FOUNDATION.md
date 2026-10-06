# TNR Research Foundation

## 1. Mục đích

Tài liệu này mô tả nền tảng nghiên cứu của **Topological Neural Routing (TNR)**.

Mục tiêu là chuyển một trực giác hình học thành một câu hỏi ML có thể kiểm chứng mà không đánh đồng:

- hình học vật thể 3D;
- topology của graph;
- topology của representation space;
- kiến trúc mạng nơ-ron;
- và cơ chế routing.

TNR không nghiên cứu việc “xếp neuron theo hình một đa diện”. Giá trị tiềm năng nằm ở **cấu trúc kết nối, dynamic routing và cách tổ chức representation/computation theo module**.

---

## 2. Điểm xuất phát: fixed layered computation

Một mạng nơ-ron feed-forward điển hình có thể được viết gần đúng:

$$
x \rightarrow L_1 \rightarrow L_2 \rightarrow L_3 \rightarrow \cdots \rightarrow L_n.
$$

Thông tin chủ yếu tiến theo chiều sâu.

Các kiến trúc hiện đại đã làm cấu trúc này linh hoạt hơn:

- **ResNet** thêm skip connections;
- **DenseNet** tạo nhiều kết nối giữa các layer;
- **Transformer** cho token tương tác rộng trong một layer qua attention;
- **Graph Neural Network** thực hiện message passing trên graph;
- **Mixture-of-Experts** dùng router để chọn expert;
- **recurrent architectures** tái sử dụng computation qua thời gian.

Vì vậy TNR không xuất phát từ giả định rằng “ML chưa từng có graph hoặc routing”.

Câu hỏi của TNR hẹp hơn:

> Có lớp bài toán nào mà việc tổ chức computation thành các module tái sử dụng, với routing phụ thuộc input trên một topology giàu kết nối, mang lại lợi ích đo được so với fixed-depth computation dưới budget tương đương?

---

## 3. Cảm hứng hình học

Nguồn cảm hứng ban đầu là một khối đa diện 8 mặt có genus 3 với global face adjacency rất dày.

Điểm cần trừu tượng hóa không phải hình dạng 3D mà là sự đồng tồn tại của:

$$
\boxed{\text{local structure}}
$$

và

$$
\boxed{\text{global adjacency}}.
$$

Mỗi đơn vị có cấu trúc nội bộ, nhưng ở cấp cao hơn các đơn vị có khả năng tương tác trực tiếp với nhiều đơn vị khác.

Trong TNR, abstraction tối thiểu là:

$$
\text{neurons/features}
\rightarrow
\text{modules}
\rightarrow
\text{module interaction graph}.
$$

Trường hợp 8 module với complete adjacency cho:

$$
G = K_8.
$$

Tuy nhiên K8 chỉ là **một experimental topology ban đầu**, không phải một dogma kiến trúc.

---

## 4. Từ stacked layers sang routed modules

Thay vì:

~~~text
Layer 1
   ↓
Layer 2
   ↓
Layer 3
   ↓
Layer 4
~~~

TNR khảo sát dạng:

~~~text
        M2 ───── M3
      ╱ │ ╲     ╱ │ ╲
    M1──┼──M4───┼──M5
      ╲ │ ╱     ╲ │ ╱
        M6 ───── M7
           ╲   ╱
             M8
~~~

Sơ đồ ASCII chỉ mang tính trực giác. Định nghĩa logic là:

$$
M_i \leftrightarrow M_j, \qquad \forall i\neq j.
$$

Điều quan trọng hơn complete adjacency là **edge activation không cố định**.

Một edge tồn tại trong topology không đồng nghĩa edge đó phải được sử dụng trong mọi computation.

---

## 5. Dynamic routing

Trạng thái module i có thể cập nhật theo:

$$
h_i^{t+1}
=
f_i
\left(
h_i^t,
\sum_{j\neq i}
g_{ij}^t W_{ij} h_j^t
\right).
$$

Trong đó:

- $f_i$: local computation của module i;
- $W_{ij}$: channel/interface từ j đến i;
- $g_{ij}^t$: routing gate tại iteration t.

Gate mềm:

$$
g_{ij}^t \in [0,1].
$$

Gate cứng:

$$
g_{ij}^t \in \{0,1\}.
$$

Nếu mọi gate luôn bật, mô hình chỉ trở thành một dense modular network.

Do đó một phiên bản TNR có ý nghĩa cần pressure buộc router lựa chọn, ví dụ:

$$
\sum_{ij} g_{ij}^t \le K,
$$

hoặc một explicit communication/computation cost:

$$
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda \mathcal{C}_{route}.
$$

Khi ấy routing trở thành một quyết định có trade-off thay vì một kết nối miễn phí.

---

## 6. Depth so với topology + routing + iterations

Layered architecture dùng depth như một tài nguyên:

$$
\text{computation}
\approx
\text{number of sequential transformations}.
$$

TNR khảo sát:

$$
\text{computation}
\approx
\text{topology}
+
\text{routing}
+
\text{iterations}.
$$

Cùng một tập module có thể được tái sử dụng:

$$
H^0
\rightarrow
H^1
\rightarrow
H^2
\rightarrow
\cdots
\rightarrow
H^T.
$$

Điều này biến computation từ một đường đi được mã hóa cứng trong kiến trúc sang một trajectory có thể thay đổi theo input.

---

## 7. Bốn động cơ chính để kiểm chứng

### 7.1. Khoảng cách truyền thông tin ngắn

Trong một stack 100 layer:

$$
d(L_1,L_{100}) \approx 99
$$

nếu chỉ xét đường tuần tự.

Trong complete module graph:

$$
d(M_i,M_j)=1.
$$

Giả thuyết cần test:

> Với task cần tích hợp thông tin giữa các subsystem khác nhau, short communication path có tạo lợi ích đo được hay không?

Không được suy ra trước rằng “ngắn hơn luôn tốt hơn”, vì dense connectivity cũng có thể làm tăng interference, optimization difficulty và compute.

### 7.2. Không bắt buộc một thứ tự xử lý cố định

Một pipeline cứng:

$$
A \rightarrow B \rightarrow C \rightarrow D
$$

không phù hợp tự nhiên với task mà dependency thay đổi theo input.

Ví dụ:

$$
A \rightarrow F \rightarrow C
$$

cho input loại 1, nhưng:

$$
A \rightarrow B \rightarrow H \rightarrow D
$$

cho input loại 2.

TNR cho phép router chọn trajectory tương ứng.

Đây là hypothesis trung tâm của chương trình.

### 7.3. Module specialization

Ví dụ minh họa:

| Module | Vai trò có thể có |
|---|---|
| M1 | perception |
| M2 | memory |
| M3 | spatial |
| M4 | symbolic |
| M5 | causal |
| M6 | retrieval |
| M7 | planning |
| M8 | decision |

TNR không giả định specialization này tồn tại sẵn.

Có ít nhất ba khả năng cần phân biệt:

1. role được thiết kế trước;
2. role xuất hiện qua training;
3. không có specialization ổn định.

Do đó specialization phải có metric và probe riêng.

### 7.4. Iterative computation

Nếu computation cần quay lại một subsystem trước khi quyết định:

$$
M_{causal}
\rightarrow
M_{memory}
\rightarrow
M_{causal}
\rightarrow
M_{decision},
$$

recurrent routed graph biểu diễn trajectory này trực tiếp hơn một fixed feed-forward path.

Nhưng việc mô hình có route lặp **không đồng nghĩa** nó đã reasoning. Claim về reasoning phải được chứng minh độc lập.

---

## 8. Hai cấp cấu trúc: local computation và global topology

TNR tách:

### Cấp 1 — Local computation

Mỗi module có state và transformation riêng:

$$
H_i = f_i(X_i).
$$

### Cấp 2 — Global interaction

Các module trao đổi thông tin qua graph:

$$
G=(V,E).
$$

Với K8:

$$
|V|=8, \qquad |E|=28
$$

nếu coi graph vô hướng đơn.

Trong dynamic directed routing, $i\rightarrow j$ và $j\rightarrow i$ có thể là hai phép biến đổi/gate khác nhau.

Điều này cho phép phân biệt ba đối tượng:

1. **available topology** — cạnh có thể dùng;
2. **activated topology** — cạnh được gate mở;
3. **actual computation trajectory** — đường computation thực sự của case.

Ba đối tượng này không được đánh đồng trong measurement.

---

## 9. Representation interference và “topological capacity”

Hình học genus 3 gợi ra một analogy: khi adjacency toàn cục khó embedding trong một surface đơn giản, cần thêm topological capacity.

Trong ML, analogy này chỉ nên dùng để **sinh hypothesis**, không phải làm bằng chứng.

Câu hỏi ML hợp lệ là:

> Liệu phân chia representation thành nhiều module với interface/routing rõ ràng có làm giảm interference hoặc tăng compositional reuse so với representation monolithic dưới cùng capacity/compute hay không?

So sánh:

$$
h \in \mathbb{R}^{d}
$$

với:

$$
H = \{H_1, H_2, ..., H_n\}.
$$

Để biến câu hỏi này thành khoa học cần operational metric cho interference, ví dụ:

- gradient conflict;
- task interference;
- representational overlap;
- catastrophic cross-task degradation;
- route instability;
- causal ablation sensitivity.

**Genus = 3 không được xem là một hyperparameter ML có ý nghĩa trước khi có derivation hoặc evidence.**

---

## 10. Prior-art boundary

Nếu claim chỉ là:

> “mọi module đều có thể kết nối với mọi module”

thì novelty rất yếu hoặc không tồn tại.

Các họ kiến trúc liên quan trực tiếp gồm:

- attention;
- dense/skip connectivity;
- graph neural networks;
- recurrent computation;
- mixture-of-experts;
- routing networks;
- conditional computation;
- modular neural networks.

Do đó TNR phải tránh novelty-by-diagram.

Một contribution có giá trị, nếu xuất hiện, nhiều khả năng phải nằm ở một hoặc nhiều điểm sau:

1. một routing mechanism có tính chất mới;
2. topology + constraint tạo behavior đo được mà baseline không có;
3. objective cost-aware cho edge activation;
4. mechanism giảm interference có causal evidence;
5. regime mà topology + iterations đạt compute/accuracy trade-off tốt hơn fixed depth;
6. formalization liên kết dependency geometry của task với routing topology;
7. result âm xác định rõ khi adaptive topology **không** có ích.

Ngay cả kết quả FAIL cũng có giá trị nếu nó loại bỏ một hypothesis đã preregistered.

---

## 11. Thí nghiệm tối thiểu đầu tiên

Không bắt đầu bằng LLM.

Mục tiêu đầu tiên là một môi trường nhỏ, reproducible và có dependency geometry biết trước.

### 11.1. Các model cần so sánh

Dưới parameter budget và compute budget tương đương:

$$
A = \text{8-layer MLP}
$$

$$
B = \text{residual/dense modular baseline}
$$

$$
C = \text{static K8 modular network}
$$

$$
D = \text{gated K8 modular network}.
$$

Có thể cần thêm recurrent hoặc generic graph-routing baseline nếu prior-art audit cho thấy đây là comparator bắt buộc.

### 11.2. Task geometry

Task family phải có dependency path thay đổi theo case.

Ví dụ synthetic generator có latent route:

~~~text
Case family A: 1 → 4 → 7
Case family B: 1 → 6 → 3
Case family C: 2 → 8 → 5 → 3
~~~

Ground-truth dependency không nhất thiết đưa cho model; có thể chỉ dùng cho evaluation.

### 11.3. Router pressure

Nếu D có thể bật mọi edge với chi phí bằng 0, experiment không kiểm tra routing.

Cần preregister ít nhất một constraint như:

$$
\sum_{ij}g_{ij}^t \le K
$$

hoặc:

$$
\mathcal{C}_{route}
=
\sum_{t,i,j}
c_{ij}g_{ij}^t.
$$

### 11.4. Metric families

**Task performance**

- accuracy / loss;
- OOD generalization nếu có.

**Compute**

- activated edges;
- module calls;
- FLOPs hoặc proxy;
- iterations.

**Routing**

- route recovery so với latent dependency;
- route entropy;
- route stability;
- unnecessary-edge rate.

**Mechanism**

- edge ablation;
- module ablation;
- route intervention;
- counterfactual dependency change.

Không được kết luận mechanism chỉ từ task accuracy.

---

## 12. Giả thuyết trung tâm đầu tiên

Dạng claim đáng kiểm chứng:

$$
\boxed{
\text{Adaptive topological routing}
>
\text{fixed depth}
}
$$

trong một miền hẹp:

> task family có **variable computational dependency graph**, dưới parameter/compute budget được khóa trước.

Dấu “>” chưa được định nghĩa cho tới khi preregistration khóa metric chính.

Nó có thể được operationalize thành một trong các dạng:

- accuracy cao hơn tại cùng compute;
- compute thấp hơn tại cùng accuracy;
- generalization tốt hơn trên unseen route composition;
- route fidelity tốt hơn mà task performance không giảm.

Không được đổi meaning của dấu “>” sau khi nhìn kết quả.

---

## 13. Outcome khoa học

### PASS

Evidence prospective hỗ trợ claim đã preregistered trong đúng phạm vi task và budget.

### FAIL

Baseline bằng hoặc tốt hơn, hoặc adaptive routing không tạo lợi ích theo metric chính.

FAIL được giữ trong lineage; không “sửa hypothesis” hồi tố để biến thành PASS.

### UNRESOLVED

Execution/measurement hợp lệ nhưng evidence không đủ phân biệt claim, hoặc preregistered validity condition không đạt.

UNRESOLVED không phải PASS.

---

## 14. Confound phải khóa trước execution

Ít nhất cần kiểm soát:

- parameter count;
- training examples/tokens;
- optimizer;
- learning-rate budget;
- compute budget;
- activation count;
- recurrent steps;
- hidden width;
- edge parameterization;
- seed policy;
- stopping rule;
- hyperparameter-search budget.

Nếu adaptive model được phép có nhiều compute hơn, kết quả không còn phân biệt được lợi ích của topology với lợi ích của budget.

---

## 15. Ablation cần thiết nếu có tín hiệu dương

Nếu gated K8 thắng, chưa thể kết luận “K8 là nguyên nhân”.

Cần tách:

$$
\text{topology effect}
$$

khỏi:

$$
\text{routing effect}
$$

và:

$$
\text{iteration/reuse effect}.
$$

Các ablation ứng viên:

- static K8 vs gated K8;
- gated sparse graph vs gated K8;
- same router trên random graph;
- same topology nhưng fixed route;
- same compute với recurrent dense baseline;
- shuffled latent dependency;
- disabled recurrence;
- shared vs unshared module weights.

Chỉ sau các phép tách này mới có thể nói về mechanism.

---

## 16. Quan hệ với CQG, ARN, ArcLLM/NEXUS và SIX

TNR được giữ độc lập để tránh contamination giữa hypothesis mới và các hệ đã có lineage riêng.

### ARN

Nếu có evidence, ARN có thể kế thừa:

- modular representation topology;
- routing substrate;
- interface giữa representation modules.

### CQG

CQG có thể kế thừa bài toán decision trên edge:

$$
\text{activate }(i\rightarrow j)?
$$

với:

$$
\text{expected utility}
-
\text{edge cost}.
$$

Từ đó có thể mở rộng từ “có đáng lấy thêm information không?” sang “có đáng thực hiện thêm computation/communication qua edge này không?”.

### ArcLLM / NEXUS

Chỉ liên quan khi cần:

- schedule routed computation;
- optimize sparse activation;
- manage state residency;
- thực thi conditional graph hiệu quả.

### SIX

Không phải dependency hiện tại.

Chỉ mở liên hệ nếu có evidence rằng topology hỗ trợ representation transfer, causal structure transfer hoặc cross-model modular reuse.

---

## 17. Provenance

TNR giữ scientific provenance độc lập.

Nếu một mechanism đủ mạnh để dùng nơi khác:

1. freeze exact evidence;
2. formalize claim boundary;
3. tạo artifact/contract kế thừa;
4. downstream repo reference exact version/hash;
5. không merge lịch sử nghiên cứu chỉ để “hội tụ code”.

Nguyên tắc:

$$
\boxed{\text{TNR discovers}}
$$

$$
\boxed{\text{downstream systems inherit}}
$$

---

## 18. Bước khoa học tiếp theo

Bước tiếp theo:

**TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT**

Mục tiêu:

- rà soát prior art;
- xác định phần nào của ý tưởng đã tồn tại;
- thu hẹp claim có thể kiểm chứng;
- chọn comparator bắt buộc;
- không train model;
- không mở outcome.

Sau P0, nếu vẫn còn research gap hợp lệ:

**TNR_P1_VARIABLE_DEPENDENCY_ROUTING_PREREGISTRATION**

P1 phải khóa prospective experiment trước implementation/execution.
