"""Plot milestone definitions directly from native world geometry."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from .milestone_env import MilestoneRawEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for stage,ax in enumerate(axes.flat[:5],1):
        env=MilestoneRawEnv(stage)
        for q in env.world.platforms:
            x,y,w,h=q.rect_at(0.,env.world.physics)
            ax.add_patch(Rectangle((x,y),w,h,facecolor='#cbd5e1' if q.kind=='normal' else '#7dd3fc',edgecolor='#64748b'))
        for d in env.world.special.devices.values():
            x,y,w,h=env.world.special.rect(d,0.)
            color='#fb7185' if d.type_id==201 else '#fbbf24' if d.type_id in (501,508) else '#67e8f9'
            ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#64748b'))
        for i,spec in enumerate(env.specs,1):
            src=env.bounds(spec.source);dst=env.bounds(spec.target)
            sx,sy=src[0]+src[2]/2,src[1]-16
            tx,ty=dst[0]+dst[2]/2,dst[1]-16
            ax.add_patch(FancyArrowPatch((sx,sy),(tx,ty),arrowstyle='->',mutation_scale=12,
                          color='#2563eb',connectionstyle=f'arc3,rad={-.2 if stage!=2 else .2}',linewidth=1.5))
            ax.text((sx+tx)/2,min(sy,ty)-30,str(i),color='#1d4ed8',ha='center',fontweight='bold',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
            ax.plot([dst[0],dst[0]+dst[2]],[dst[1]-3]*2,color='#059669',lw=3)
        gx,gy,gw,gh=env.level.goal
        ax.add_patch(Rectangle((gx,gy),gw,gh,facecolor='#a7f3d0',edgecolor='#047857'))
        ax.scatter(*env.level.spawn,c='#0f172a',s=30,zorder=5)
        ax.set(xlim=(0,env.level.width),ylim=(env.level.height,0),aspect='equal',
               title=f'튜토리얼 {stage} · 핵심 구간 {len(env.specs)}개')
        ax.set_xlabel('실제 월드 x (px)');ax.set_ylabel('실제 월드 y (px)');ax.grid(alpha=.15)
        env.close()
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.95,'검은 점: 원래 출발점\n녹색 상자: 이글루\n파란 화살표: 핵심 구간의 특수 블록 → 목표 그룹\n초록 선: 다음 목표 발판의 윗면\n\n화살표를 따라 이동하도록 제어하지 않습니다.\n실제 착지 이벤트가 순서대로 발생해야 +10입니다.\n이동 발판은 0초의 위치만 그렸으며,\n보상 판정은 이동하는 실제 발판 ID를 추적합니다.\n\n좌표는 화면 배율 적용 전 게임 월드 기준입니다.',va='top',linespacing=1.6)
    fig.suptitle('사용자 보상에 사용한 핵심 구간 정의',fontsize=17)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=150);plt.close(fig)


if __name__=='__main__':main()
