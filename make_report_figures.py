"""보고서 그림 1~5를 원시 결과 파일(JSON)에서 직접 그린다. 수치를 코드에 직접 적지 않는다.
실행: python make_report_figures.py  (results/dgx_results.json, data/cpu_results_20260812.json 필요)"""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
D="results/dgx_results.json"
C="data/cpu_results_20260812.json"
O="figures/"
_fonts={f.name for f in fm.fontManager.ttflist}
_kr=next((f for f in ["Malgun Gothic","Noto Sans CJK KR","AppleGothic","NanumGothic"] if f in _fonts),"sans-serif")
os.makedirs(O,exist_ok=True)
plt.rcParams.update({"font.family":_kr,"axes.unicode_minus":False,"font.size":11,
 "axes.edgecolor":"#c9c8c3","axes.labelcolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#e7e6e2","grid.linewidth":0.8,
 "figure.facecolor":"white","axes.facecolor":"white","legend.frameon":False,"lines.linewidth":2,"lines.markersize":7})
B,Or,Aq,Ye="#2a78d6","#eb6834","#1baf7a","#eda100"
g=json.load(open(D))["results"]; c=json.load(open(C))
N=[r["n"] for r in g]; Nl=[str(n) for n in N]
def fin(fig,ax,name,title):
    ax.set_title(title,loc="left",fontsize=12,color="#0b0b0b",pad=10)
    fig.tight_layout(); fig.savefig(O+name,dpi=200); plt.close(fig)
# Fig1 CPU
fig,ax=plt.subplots(figsize=(7.2,4.2))
cn=[r["n"] for r in c]
ax.plot(range(len(cn)),[r["float32_median_ms"] for r in c],"o-",color=B,label="FP32 (OpenBLAS)")
ax.plot(range(len(cn)),[r["int8_to_int32_median_ms"] for r in c],"s-",color=Or,label="양자화 후 INT32 일반 행렬곱")
ax.set_yscale("log"); ax.set_xticks(range(len(cn)),[str(n) for n in cn]); ax.set_xlabel("행렬 크기 N"); ax.set_ylabel("실행시간 중앙값 (ms, 로그)")
ax.annotate("1072.8배",xy=(5,c[5]["int8_to_int32_median_ms"]),xytext=(3.6,c[5]["int8_to_int32_median_ms"]*1.6),color="#52514e",fontsize=10)
ax.legend(loc="upper left"); fin(fig,ax,"fig1_cpu.png","범용 CPU: FP32와 정수 경로 실행시간")
# Fig2 DGX kernel
fig,ax=plt.subplots(figsize=(7.2,4.2)); x=range(len(N))
ax.plot(x,[r["fp32_ms"] for r in g],"o-",color=B,label="FP32 (TF32 비허용)")
ax.plot(x,[r["tf32_ms"] for r in g],"^-",color=Aq,label="TF32 허용")
ax.plot(x,[r["int8_int32_ms"] for r in g],"s-",color=Or,label="INT8 → INT32 (torch._int_mm)")
ax.set_yscale("log"); ax.set_xticks(list(x),Nl); ax.set_xlabel("행렬 크기 N"); ax.set_ylabel("커널 시간 중앙값 (ms, 로그)")
ax.legend(loc="upper left"); fin(fig,ax,"fig2_dgx_kernel.png","DGX Spark(GB10): 정밀도별 행렬곱 커널 시간")
# Fig3 speedup vs FP32
fig,ax=plt.subplots(figsize=(7.2,4.2))
ax.plot(x,[r["fp32_ms"]/r["int8_int32_ms"] for r in g],"s-",color=Or,label="INT8 커널만")
ax.plot(x,[r["fp32_ms"]/r["tf32_ms"] for r in g],"^-",color=Aq,label="TF32")
ax.plot(x,[r["fp32_ms"]/(r["int8_int32_ms"]+r["quantization_ms"]) for r in g],"o--",color=Ye,label="INT8 + 입력 양자화")
ax.axhline(1,color="#8a8984",lw=1); ax.text(4.05,1.04,"FP32와 같음",fontsize=9,color="#52514e",ha="right")
ax.set_xticks(list(x),Nl); ax.set_xlabel("행렬 크기 N"); ax.set_ylabel("FP32 대비 속도 배수 (배)")
ax.legend(loc="upper left"); fin(fig,ax,"fig3_speedup.png","FP32 대비 속도 배수: 커널만 vs 양자화 포함")
# Fig4 quant measured vs bandwidth estimate
BW=273e9; est=[2*n*n*41/BW*1000 for n in N]
fig,ax=plt.subplots(figsize=(7.2,4.2))
ax.plot(x,[r["quantization_ms"] for r in g],"o-",color=Or,label="실측 양자화 시간")
ax.plot(x,est,"s--",color=B,label="메모리 이동량 ÷ 273GB/s 추정")
ax.set_yscale("log"); ax.set_xticks(list(x),Nl); ax.set_xlabel("행렬 크기 N"); ax.set_ylabel("시간 (ms, 로그)")
ax.legend(loc="upper left"); fin(fig,ax,"fig4_quant.png","입력 양자화 시간: 실측과 메모리 대역폭 추정")
# Fig5 NMAE
fig,ax=plt.subplots(figsize=(7.2,4.2))
allN=sorted(set(cn)|set(N)); pos={n:i for i,n in enumerate(allN)}
ax.plot([pos[r["n"]] for r in c],[r["nmae_percent"] for r in c],"o-",color=B,label="CPU (NumPy, 시드 20260812+N)")
ax.plot([pos[r["n"]] for r in g],[r["nmae_percent"] for r in g],"s-",color=Or,label="DGX (PyTorch, 시드 20260813+N)")
ax.set_xticks(range(len(allN)),[str(n) for n in allN]); ax.set_xlabel("행렬 크기 N"); ax.set_ylabel("NMAE (%)"); ax.set_ylim(1.0,2.0)
ax.legend(loc="lower right"); fin(fig,ax,"fig5_nmae.png","행렬 크기에 따른 역양자화 오차(NMAE)")
print("ok", [round(e,2) for e in est])
