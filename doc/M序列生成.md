# 5G NR PSS与PRS定位参考信号中M序列生成规范

在5G NR标准中，**PSS（主同步信号）**的生成直接采用了一个长度为127的M序列（m-sequence），而**定位功能（Positioning）**主要依赖于**PRS（定位参考信号）**，其底层使用的是由两个不同M序列组合而成的伪随机序列（长度为31的Gold序列）[1-3]。

以下为您完整展示这两类信号中有关M序列生成的具体数学定义与相关信息：

## 1. PSS（主同步信号）中的M序列生成

PSS的生成直接依赖于一个本原多项式生成的M序列 $x(i)$ [2]。

**对于下行链路主同步信号（Downlink PSS）：**
* PSS序列 $d_{\text{PSS}}(n)$ 的生成公式为：$d_{\text{PSS}}(n) = 1 - 2x(m)$ [2]。
* 其中偏移量 $m$ 的定义为：$m = (n + 43N_{ID}^{(2)}) \bmod 127$ [2]。
* 序列长度满足：$0 \le n < 127$ [2]。
* **核心M序列递推多项式**为：$x(i+7) = (x(i+4) + x(i)) \bmod 2$ [2]。
* **M序列的初始化**设定为：$\left[x(6) \quad x(5) \quad x(4) \quad x(3) \quad x(2) \quad x(1) \quad x(0)\right] = \left[1 \quad 1 \quad 1 \quad 0 \quad 1 \quad 1 \quad 0\right]$ [2]。

**对于侧行链路主同步信号（Sidelink S-PSS）：**
* S-PSS序列 $d_{\text{S-PSS}}(n)$ 的生成公式为：$d_{\text{S-PSS}}(n) = 1 - 2x(m)$ [4]。
* 偏移量 $m$ 与下行链路略有不同，定义为：$m = (n + 22 + 43N_{ID, 2}^{SL}) \bmod 127$ [4]。
* 序列长度同样满足 $0 \le n < 127$ [4]。
* 核心M序列递推公式与初始化过程与上述下行链路PSS完全相同 [4]。

## 2. 定位参考信号（PRS）底层的M序列（伪随机序列）生成

定位参考信号（PRS）所使用的序列属于通用的伪随机序列（Pseudo-random sequence），该伪随机序列是由**两个长度为31的M序列复合而成的Gold序列** [1]。

**通用伪随机序列生成过程：**
* 输出序列 $c(n)$ 定义为：$c(n) = (x_1(n+N_C) + x_2(n+N_C)) \bmod 2$ [1]。
* 其中常数 $N_C = 1600$ [1]。
* **第一个M序列 $x_1$** 的递推公式为：$x_1(n+31) = (x_1(n+3) + x_1(n)) \bmod 2$ [1]。
  * $x_1$ 的初始化固定为：$x_1(0)=1$，且对于 $n=1,2,\dots,30$，有 $x_1(n)=0$ [1]。
* **第二个M序列 $x_2$** 的递推公式为：$x_2(n+31) = (x_2(n+3) + x_2(n+2) + x_2(n+1) + x_2(n)) \bmod 2$ [1]。
  * $x_2$ 的初始化由具体应用的初始值 $c_{\text{init}}$ 决定，满足：$c_{\text{init}} = \sum_{i=0}^{30} x_2(i) \cdot 2^i$ [1]。

## 3. PRS（定位参考信号）对M序列的调用与初始化

在生成了上述基础的伪随机序列 $c(n)$ 后，网络会使用它来调制生成用于定位的复值参考信号 $r(m)$。

**下行链路定位参考信号（Downlink PRS）：**
* 序列映射公式为：$r(m) = \frac{1}{\sqrt{2}}(1 - 2c(2m)) + j\frac{1}{\sqrt{2}}(1 - 2c(2m+1))$ [3]。
* 针对第二个M序列 $x_2$ 的初始化参数 $c_{\text{init}}$ 的计算方式为：
  $c_{\text{init}} = \left( 2^{22} \lfloor \frac{n_{ID,seq}^{PRS}}{1024} \rfloor + 2^{10}(N_{symb}^{slot}n_{s,f}^{\mu} + l + 1)(2(n_{ID,seq}^{PRS} \bmod 1024) + 1) + (n_{ID,seq}^{PRS} \bmod 1024) \right) \bmod 2^{31}$ [5]。
* 其中 $n_{ID,seq}^{PRS} \in \{0, 1, \dots, 4095\}$ 代表下行链路PRS的序列ID [5]。

**侧行链路定位参考信号（Sidelink SL PRS）：**
* 序列映射公式同样为：$r(m) = \frac{1}{\sqrt{2}}(1 - 2c(2m)) + j\frac{1}{\sqrt{2}}(1 - 2c(2m+1))$ [6]。
* 针对第二个M序列的初始化参数 $c_{\text{init}}$ 计算方式为：
  $c_{\text{init}} = \left( 2^{22} \lfloor \frac{n_{ID,seq}^{SL-PRS}}{1024} \rfloor + 2^{10}(N_{symb}^{slot}n_{s,f}^{\mu} + l + 1)(2(n_{ID,seq}^{SL-PRS} \bmod 1024) + 1) + (n_{ID,seq}^{SL-PRS} \bmod 1024) \right) \bmod 2^{31}$ [6]。
* 其中 $n_{ID,seq}^{SL-PRS} \in \{0, 1, \dots, 4095\}$ 代表侧行链路PRS的序列ID [6]。