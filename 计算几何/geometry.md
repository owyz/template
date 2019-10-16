# 凸包（含极角排序）

```cpp
#include<cmath>
#include<cstdio>
#include<algorithm>
using namespace std;
int n;
struct node{
	double x;
	double y;
}p[100010];
inline double X(node a,node b,node c)//叉积 
{
	double x1=b.x-a.x,y1=b.y-a.y;
	double x2=c.x-a.x,y2=c.y-a.y;
	return x1*y2-x2*y1;
}
inline double len(node a,node b)
{return sqrt( (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y) );}//勾股 
inline int cmp(const node & a,const node & b)
{
	double pp=X(p[1],a,b);
	if(pp>0) return 1;
	if(pp<0) return 0;
	return len(p[1],a)<len(p[1],b);
}

node s[100010]={0};
int top=0;

int main()
{
	scanf("%d",&n);
	if(n==1||n==0){printf("0");return 0;}
	for(int i=1;i<=n;i++)//输入 
	{
		scanf("%lf%lf",&p[i].x,&p[i].y);
		if(p[i].y<p[1].y)
			swap(p[1],p[i]);
		else if(fabs(p[i].y-p[1].y)<1e-5&&p[i].x<p[1].x)
			swap(p[1],p[i]);
	}
	if(n==2){printf("%lf",len(p[1],p[2]));return 0;}
	for(int i=2;i<=n;i++)//挪原点 
	{
		p[i].x-=p[1].x;
		p[i].y-=p[1].y;
	}
	p[1]={0.0,0.0};
	sort(p+2,p+1+n,cmp);
	s[top++]=p[1];
	s[top++]=p[2];
	p[++n]=p[1];
	for(int i=3;i<=n;i++)//求凸包
	{
		while(top!=0&&X(s[top-2],s[top-1],p[i])<0)top--;
		s[top++]=p[i];
	}
	double ans=0;
	//for(int i=1;i<=n;i++) printf("^%lf^%lf^\n",p[i].x,p[i].y);
	do//求周长 
	{
		ans+=len(s[top-1],s[top-2]);
		top--;
	}while(top);
	printf("%.2lf\n",ans);
	return 0;
}
```