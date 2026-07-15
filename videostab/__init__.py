"""videostab: 离线视频防抖 (DUT 框架 x NLNL 改进).

数据约定 (全局统一):
- 网格顶点: (GH, GW) 个顶点, 坐标定义在代理分辨率 (proxy) 像素系.
- 顶点运动场 m_t: ndarray (GH, GW, 2), 帧 t -> t+1 背景在顶点处的运动 [dx, dy].
- 相机路径 C: ndarray (T, GH, GW, 2), C[0]=0, C[t] = sum(m_0..m_{t-1}).
- 平滑路径 P 同形状; 校正场 B_t = P_t - C_t, 渲染时对帧 t 内容平移 B_t.
"""

__version__ = "0.1.0"
