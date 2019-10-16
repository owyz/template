## st表

```cpp
#include<cmath>
int n,st[maxn][20];
int lg[maxn];//自己求
void init()
{
    for(int i=0;i<n;i++)
        scanf("%d",&st[i][0]);
    for(int i=1;(1<<i)<=n;i++)
    {
        int len=(1<<i);
        for(int bg=0;bg+len-1<n;bg++)
            st[bg][i]=max(st[bg][i-1],st[bg+(len>>1)][i-1]);
    }
}
int max(int l,int r){
    return std::max(st[a][lg[b-a+1]],st[b-(1<<lg[b-a+1])+1][lg[b-a+1]]);
}
```



## zkw线段树

```c++
// 1 - sz-1 祖先结点
// sz+1 - sz*2-2 原始数组
const int maxsz=1<<17;//131072(input len max to 131070)
struct Node{int max,sum;}tree[maxsz*2];
int n,sz;//ori_n && real_n
inline void make_tree()
{
    for(sz=(n+1)<<1;sz!=(sz&-sz);sz-=(sz&-sz));
    for(int i=sz-1;i>0;i--)
    {
        tree[i].max=max(tree[i<<1].max,tree[(i<<1)+1].max);
        tree[i].sum=tree[i<<1].sum+tree[(i<<1)+1].sum;
    }
}
inline int query_sum(int l,int r)
{
    int ans=0;
    for(int i=l-1+sz,j=r+1+sz;i^j^1;i>>=1,j>>=1)
    {
        if(~i&1)ans+=tree[i^1].sum;
        if(j&1)ans+=tree[j^1].sum;
    }
    return ans;
}
inline void change(int i,int val)
{
    for(tree[i+=sz]={val,val},i>>=1;i>0;i>>=1)
    {
        tree[i].max=max(tree[i<<1].max,tree[(i<<1)+1].max);
        tree[i].sum=tree[i<<1].sum+tree[(i<<1)+1].sum;
    }
}
```

### 可持久化线段树

```cpp
//区间第k大
int root[maxn];
struct Node{int lc,rc,num;}tree[maxn*20];
int tcnt;

int main()
{
    //init
    tcnt=1;
    root[0]=0,tree[0]={0,0,0};//save node[0]

    for(int i=1;i<=n;i++){
        root[i]=root[i-1];
        insert(a[i],root[i],1,nn);
    }
}

void insert(int num,int & rt,int l,int r)
{
    /******************/
    tree[tcnt]=tree[rt];
    tree[tcnt].sz++;//modify this node
    rt=tcnt++;
    /******************/
    if(l==r)return;
    
    int mid=(l+r)>>1;
    if(num<=mid) 
        insert(num,tree[rt].lc,l,mid);
    else 
        insert(num,tree[rt].rc,mid+1,r);
}

int query(int rt1,int rt2,int k,int l,int r)
{
    if(l==r)return l;
    int leftnum=tree[tree[rt2].lc].sz-tree[tree[rt1].lc].sz;
    int mid=(l+r)>>1;
    if(leftnum<k)
        return query(tree[rt1].rc,tree[rt2].rc,k-leftnum,mid+1,r);
    else
        return query(tree[rt1].lc,tree[rt2].lc,k,l,mid);
}
```



## 左偏树

```cpp
//一开始有n只孤独的猴子，然后他们要打m次架，每次打架呢，都会拉上自己朋友最牛叉的出来跟别人打，打完之后战斗力就会减半，每次打完架就会成为朋友（正所谓不打不相识o(∩_∩)o ）。问每次打完架之后那俩猴子最牛叉的朋友战斗力还有多少，若朋友打架就输出-1
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
int n,m;
struct ltree{
    int l;
    int r;
    int dist;
    int fa;
    int key;
}t[100010];
int find(int x)
{
    return (x==t[x].fa)?x:(find(t[x].fa));
}
int merge(int x,int y)
{
    if(!x) return y;
    if(!y) return x;
    if(t[x].key<t[y].key) swap(x,y);
    t[x].r=merge(t[x].r,y);
    t[t[x].r].fa=x;
    if(t[t[x].l].dist<t[t[x].r].dist) swap(t[x].l,t[x].r);
    t[x].dist=t[t[x].r].dist+1;
    return x;
}
int pop(int x)
{
    int l=t[x].l,r=t[x].r;
    t[l].fa=l;
    t[r].fa=r;
    t[x].l=t[x].r=t[x].dist=0;
    return merge(l,r);
}
void init()
{
	memset(t,0,sizeof(t));
    t[0].dist=-1;
    for(int i=1;i<=n;i++)
    {
        t[i].fa=i;
        scanf("%d",&t[i].key);
    }
}
int main()
{
    //freopen("test.in","r",stdin);
    while(1==scanf("%d",&n))
    {
        init();
        scanf("%d",&m);
        for(int x,y,i=0;i<m;i++)
        {
			scanf("%d%d",&x,&y);
			x=find(x);
			y=find(y);
			if(x==y)
			{
			   printf("-1\n");
			   continue;
			}
			t[x].key>>=1;
			t[y].key>>=1;
			int x_son=pop(x),y_son=pop(y);
			x_son=merge(x_son,x);
			y_son=merge(y_son,y);
			x_son=merge(x_son,y_son);
			printf("%d\n",t[x_son].key);
        }
    }
    return 0;
}
```



## 二维树状数组

```cpp
#include<stdio.h>
//矩阵单点修改、矩形区域查询
const long long MOD=1e9+7;

inline int lowbit(int x){return x&-x;}

int n,m;
long long c[1010][1010];

void update(int x,int y,long long k)
{
	int yy=y;
	while(x<=n)
	{
		y=yy;
		while(y<=n)
		{
			c[x][y]+=k;
			c[x][y]%=MOD;
			y+=lowbit(y);
		}
		x+=lowbit(x);
	}
}
long long que(int x,int y)
{
	int yy=y;
	long long ans=0;
	while(x)
	{
		y=yy;
		while(y)
		{
			ans+=c[x][y];
			ans%=MOD;
			y-=lowbit(y);
		}
		x-=lowbit(x);
	}
	return ans;
}

int main()
{
	scanf("%d%d",&n,&m);
	while(m--)
	{
		char opt[4];
		int x,y,a,b;
		scanf("\n%s",opt);
		if(opt[0]=='A')
		{
			scanf("%d%d%d",&x,&y,&a);
			x++,y++;
			update(x,y,a);
		}
		else
		{
			scanf("%d%d%d%d",&x,&y,&a,&b);
			a++,b++;
			printf("%lld\n",((que(a,b)-que(a,y)-que(x,b)+que(x,y))%MOD+MOD)%MOD);
		}
	}
	return 0;
}
```

