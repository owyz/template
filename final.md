# 字符串

## kmp

1. 求next数组

   ```cpp
   int nex[maxp];
   char p[maxp],s[maxs];
   int lenp,lens;
   void getnext()
   {
       nex[0]=-1;nex[1]=0;
       for(int i=1,j=0;i<lenp;)
       {
           if(p[i]==p[j] || j<0)
           {
               i++;j++;
               //if(p[i]==p[j]) //普通版本多匹配的原因是p[j]==p[nex[j]]，但这种方法不支持循环节
                   //nex[i]=nex[j];
               //else
                   nex[i]=j;
           }
           else
               j=nex[j];
       }
   }
   ```

2. kmp

   ```cpp
   lenp=strlen(p);
   lens=strlen(s);
   int kmp()
   {
       getnext();
       for(int i=0,j=0;i<lens;)
       {
           if(s[i]==p[j] || j<0){
               i++;j++;
               if(j==lenp){
                   //1
                   printf("%d\n",i-j);
                   j=nex[j];
                   //2 
                   //return i-j;
               }
           }
           else j=nex[j];
       }
       return -1;
   }
   ```

   

3. kmp求循环节

   `0 - i-1`的最大循环节为`i-next[i]`

## Manacher

```cpp
#include<cstring>
#include<algorithm>
char str[maxn*2+5]; // #s[0]#s[1]#s[2]#s[3]#
int lr[maxn*2+5]; // 左右长度&&原串回文长度
int maxr,mid; // 最右,对应的中点
int maxlen; // 最长回文长度

int Manacher()
{
    memset(lr,0,sizeof(lr));
    maxr=mid=maxlen=0;
    int len=strlen(str),Len=len*2+1;

    for(int i=Len-1;i>=0;i--){
        if(i%2) str[i]=str[i/2];
        else str[i]='#';
    }

    for(int i=0;i<Len;i++)
    {
        if(i<maxr) lr[i]=std::min(lr[2*mid-i],maxr-i);
        while(i-lr[i]-1>=0 && i+lr[i]+1<Len && str[i-lr[i]-1]==str[i+lr[i]+1]) lr[i]++;
        if(lr[i]+i>maxr){
            maxr=lr[i]+i;
            mid=i;
        }

        maxlen=std::max(maxlen,lr[i]);
    }
    return maxlen;
}
```

# 数据结构

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

## 可持久化线段树

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

# 图论

## 最短路

### Dijkstra

```cpp
//O(M+NlogN)
//利用未访问的、离源点最近的点更新其他所有点
#include<cstdio>
#include<cstring>
#include<queue>
using namespace std;
const int maxn=1e5+5;
const int maxm=2e5+5;

struct Edge{
    int v,nex;
    long long w;
}edge[maxm*2];
int head[maxn],ecnt;
void addedge(int u,int v,long long w){
    edge[ecnt]=(Edge){v,head[u],w};
    head[u]=ecnt++;
}
void initedge(){
    memset(head,-1,sizeof(head));
    ecnt=0;
}

long long dis[maxn];
int vis[maxn];
typedef pair<long long,int> Node;//dis,point;
priority_queue<Node,vector<Node>,greater<Node> >q;
void dij(int s)
{
    memset(dis,0x3f,sizeof(dis));
    memset(vis,0,sizeof(vis));

    for(dis[s]=0,q.push(make_pair(dis[s],s));!q.empty();)
    {
        int u=q.top().second,v;
        q.pop();
        if(vis[u])continue;
        for(int i=head[u];i!=-1;i=edge[i].nex)
        {
            v=edge[i].v;
            if(!vis[v] && dis[u]+edge[i].w<dis[v])
            {
                dis[v]=dis[u]+edge[i].w;
                q.push(make_pair(dis[v],v));
            }
        }
        vis[u]=true;
    }
}

int main()
{
    initedge();

    int n,m,s;
    scanf("%d%d%d",&n,&m,&s);
    for(long long u,v,w;m--;){
        scanf("%lld%lld%lld",&u,&v,&w);
        addedge(u,v,w);
    }

    dij(s);
    for(int i=1;i<=n;i++)
        printf("%lld%c",dis[i],i==n?'\n':' ');

    return 0;
}
```

### Bellman-Ford

```cpp
//O(NM)
//不断松弛，类似建树过程，每次至少有一点确定下来，所以至多n-1轮
const long long inf=0x3f3f3f3f3f3f3f3f;
long long dis[maxn];
void bf(int n,int s)
{
    memset(dis,0x3f,sizeof(dis));
    
    dis[s]=0;
    for(int t=n-1;t--;)
    {
        bool flag=false;
        for(int u=1,v;u<=n;u++)
        {
            if(dis[u]>=inf)continue;
            for(int i=head[u];i!=-1;i=edge[i].nex)
            {
                v=edge[i].v;
                if(dis[v]>dis[u]+edge[i].w)
                {
                    dis[v]=dis[u]+edge[i].w;
                    flag=true;
                }
            }
        }
        if(!flag)break;
    }
}
```

### spfa

```cpp
//O(NM),平均O(KM) (K<=2)
//删去与树根不连通的松弛
#include<deque>
long long dis[maxn];
int inQueue[maxn];
deque<int>q;
void spfa(int s)
{
    memset(dis,0x3f,sizeof(dis));
    memset(inQueue,0,sizeof(inQueue));
    dis[s]=0;
    for(q.push_back(s);!q.empty();)
    {
        int u=q.front(),v;
        q.pop_front();
        for(int i=head[u];i!=-1;i=edge[i].nex)
        {
            v=edge[i].v;
            if(dis[u]+edge[i].w<dis[v])
            {
                dis[v]=dis[u]+edge[i].w;
                if(!inQueue[v])
                {
                    if(dis[v]<dis[q.front()])//slf
                        q.push_front(v);
                    else
                        q.push_back(v);
                    inQueue[v]=true;
                }
            }
        }
        inQueue[u]=false;
    }
}
```

### floyd

```cpp
 void floyd(){
	int MinCost = inf;
	for(int k=1;k<=n;k++)
    {
	    for(int i=1;i<k;i++)
	        for(int j=i+1;j<k;j++)
	            MinCost = min(MinCost,dis[i][j]+mp[i][k]+mp[k][j]);//更新k点之前枚举ij求经过ijk的最小环
	    for(int i=1;i<=n;i++)
	        for(int j=1;j<=n;j++)
	            dis[i][j]=min(dis[i][j],dis[i][k]+dis[k][j]);      //松弛k点
	}
	if(MinCost==inf)puts("It's impossible.");
	else printf("%d\n",MinCost);
}
```

### 差分约束

1. 要满足所有要求 即 对于任意连边都不存在松弛的可能

   $\large x_i-x_j\le w_{ij}\iff dis[j]+w_{ji}\ge dis[i] \iff 最短路$

2. $\large反向连边\iff最长路$
   
3. $\large存在负权环||无法到达\iff不存在x_t-x_s的最大值$

4. 整数域：$\large A-B<C\iff A-B\le C-1$

### k短路

```cpp
#include<queue>
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;

int n,m,ans=0;
double E;

struct Edge{
    int next,to;
    double w;
}fe[200010],e[200010];//e存正向图，fe存反向图
int head[200010]={0},cnt=1,fhead[200010]={0},fcnt=1;//带f的都用于存反向图
void add(int u,int v,double w)
{
    e[cnt].to=v;
    e[cnt].w=w;
    e[cnt].next=head[u];
    head[u]=cnt++;//给正向图加边

    fe[fcnt].to=u;
    fe[fcnt].w=w;
    fe[fcnt].next=fhead[v];
    fhead[v]=fcnt++;//给反向图加边
}

double dis[5010];//裸spfa
bool inq[5010]={0};
void spfa()
{
    for(int i=0;i<5010;i++) dis[i]=999999999.0;
    dis[n]=0.0;
    queue<int> q;
    q.push(n);
    inq[n]=1;
    while(!q.empty())
    {
        int u=q.front();
        q.pop();
        inq[u]=0;
        for(int i=fhead[u];i;i=fe[i].next)
        {
            int v=fe[i].to;
            double w=fe[i].w;
            if(dis[v]>dis[u]+w)
            {
                dis[v]=dis[u]+w;
                if(!inq[v])
                {
                    inq[v]=1;
                    q.push(v);
                }
            }
        }
    }
}


struct Heap{
    double d;
    int u;
    bool operator > (const Heap &a)const{
        return d>a.d;
    }
}heap[2000010],temp;//手敲优先队列
int sz=1;//优先队列内元素数量+1，个人比较喜欢这种表示方法
void pop()//删除堆顶，取出堆顶直接用heap[1]即可，我没写在pop()里
{
    sz--;
    heap[1]=heap[sz];
    heap[sz]={0,0};
    int the=1,son=2;
    while(son<sz)
    {
        if(heap[son]>heap[son+1]&&son+1<sz) son++;
        if(heap[the]>heap[son]) swap(heap[the],heap[son]);
        else break;
        the=son;
        son=the<<1;
    }
}
void push(double dd,int uu)//加入一个元素，dd=f(uu)，dd、uu防止变量名冲突
{
    heap[sz]={dd,uu};
    int the=sz++,fa=the>>1;
    while(fa)
    {
        if(heap[fa]>heap[the]) swap(heap[the],heap[fa]);
        else break;
        the=fa;
        fa>>=1;
    }
}

void astar()
{
    push(dis[1],1);
    while(sz>1)
    {
        int u=heap[1].u;
        double dist=heap[1].d;//取出堆顶
        pop();//删除堆顶
        if(u==n)//n点出队，说明找到一条k短路
        {
            E-=dist;
            if(E>=1e-6) ans++;
            else return;
            continue;
        }
        for(int i=head[u];i;i=e[i].next)//拓展与u相连的节点
        {
            int v=e[i].to;
            double w=e[i].w;
            push(dist-dis[u]+w+dis[v],v);
        }
    }
}

int main()
{
    //freopen("test.in","r",stdin);
    scanf("%d%d%lf",&n,&m,&E);
    for(int i=1,u,v;i<=m;i++)
    {
        double w;
        scanf("%d%d%lf",&u,&v,&w);
        add(u,v,w);
    }
    spfa();
    astar();
    printf("%d\n",ans);
    return 0;
}
```



## 网络流

### dinic

#### 普通dinic

```cpp
#include<queue>
#include<cstdio>
#include<cstring>
#include<algorithm>

int n,m,s,t;
struct Edge{
	int nxt,to,flow;
}e[200010];
int head[100010]={0},cnt=2;
void add(int u,int v,int f)
{
	e[cnt]={head[u],v,f};
	head[u]=cnt++;
}

int dis[100010];
bool bfs()
{
	memset(dis,0,sizeof(dis));
	std::queue<int> q;
	q.push(s);
	dis[s]=1;
	while(!q.empty())
	{
		int u=q.front();
		q.pop();
		for(int i=head[u];i;i=e[i].nxt)
		{
			int v=e[i].to;
			if(dis[v]||(!e[i].flow)) continue;
			dis[v]=dis[u]+1;
			q.push(v);
		}
	}
	return dis[t]!=0;
}
int dfs(int u,int f)
{
	if(u==t||f==0)return f;
	int flow_sum=0;
	for(int i=head[u];i;i=e[i].nxt)
	{
		int v=e[i].to;
		if(dis[v]!=dis[u]+1||!e[i].flow) continue;
		int temp=dfs(v,std::min(f-flow_sum,e[i].flow));
		e[i].flow-=temp;
		e[i^1].flow+=temp;
		flow_sum+=temp;
		if(flow_sum>=f) break;
	}
	if(!flow_sum) dis[u]=-1;
	return flow_sum;
}
int dinic()
{
	int ans=0;
	while(bfs())
		while(int temp=dfs(s,0x7fffffff))
			ans+=temp;
	return ans;
}


int main()
{
	scanf("%d%d%d%d",&n,&m,&s,&t);
	for(int i=1,u,v,w;i<=m;i++)
	{
		scanf("%d%d%d",&u,&v,&w);
		add(u,v,w);
		add(v,u,0);
	}
	printf("%d\n",dinic());
	return 0;
}
```

#### 当前弧优化
```cpp
//O(N^2*M),二分图O(sqrt(N)*M)
namespace NetFlow{
    struct Edge{int v,next;ll w;}edge[maxm*2];
    int head[maxn],cnt;
    int cur[maxn];//当前弧

    int deep[maxn];
    std::queue<int>q;

    inline void bfs(int s);
    ll dfs(int u,ll maxflow);
}

int main()
{
    memset(head,-1,sizeof(head));
    cnt=0;
    /*input*/
    ll ans=0;
    int s=1,t=n;
    do{
        NetFlow::bfs(s);
        while(ll tmp=NetFlow::dfs(s,inf))//!!
            ans+=tmp;
    }while(NetFlow::deep[t]!=inf);
    cout<<ans<<endl;

    return 0;
}

inline void NetFlow::bfs(int s)
{
    memset(deep,0x3f,sizeof(deep));
    deep[s]=0;
    for(q.push(s);!q.empty();q.pop())
    {
        int u=q.front(),v;
        for(int i=head[u];i!=-1;i=edge[i].next)
        {
            v=edge[i].v;
            if(edge[i].w>0 && deep[v]==inf)
            {
                deep[v]=deep[u]+1;
                q.push(v);
            }
        }
    }
    for(int i=1;i<=n;i++)//当前弧优化
        cur[i]=head[i];
}
ll NetFlow::dfs(int u,ll maxflow)
{
    if(u==n || maxflow==0/*优化1*/)return maxflow;
    ll ans=0;
    // for(int i=head[u];i!=-1;i=edge[i].next)
    for(int& i=cur[u];i!=-1;i=edge[i].next)//当前弧优化
    {
        v=edge[i].v;
        if(deep[v]==deep[u]+1)
        {
            ll tmp=dfs(v,min(maxflow,edge[i].w));
            edge[i].w-=tmp;
            edge[i^1].w+=tmp;
            ans+=tmp;
            maxflow-=tmp;
            if(!maxflow)break;
        }
    }
    if(!ans)deep[u]=-1;//优化2
    return ans;
}
```

### 费用流

```cpp
/*输入的第一行包含四个正整数N、M、S、T，分别表示点的个数、有向边的个数、源点序号、汇点序号。
接下来M行每行包含四个正整数ui、vi、wi、fi，表示第i条有向边从ui出发，到达vi，边权为wi（即该边最大流量为wi），单位流量的费用为fi。*/
#include<queue>
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
typedef pair<int,int> Pair;
int n,m,s,t;

struct edge{
    int next,to,flow,cost;
}e[100010]={0};
int head[5010]={0},cnt=2;
inline void add(int u,int v,int f,int c)
{
    e[cnt]={head[u],v,f,c};
    head[u]=cnt++;
}

int dis[5010]={0},pe[5010]={0},pp[5010]={0};
bool inq[5010]={0};
bool spfa()
{
    queue<int> q;//可优化一半时间
    memset(dis,0x7f,sizeof(dis));
    memset(inq,0,sizeof(inq));
    q.push(s);inq[s]=1;dis[s]=0;
    while(!q.empty())
    {
        int u=q.front();
        q.pop();
        inq[u]=0;
        for(int i=head[u];i>1;i=e[i].next)
        {
            if(e[i].flow<=0) continue;
            int v=e[i].to;
            if(dis[v]>dis[u]+e[i].cost)
            {
                dis[v]=dis[u]+e[i].cost;
                pe[v]=i;
                pp[v]=u;
                if(!inq[v])
                {
                    q.push(v);
                    inq[v]=1;
                }
            }
        }
    }
    return dis[t]<0x7f7f7f7f;
}

Pair work()
{
    int F=0,C=0;
    while(spfa())
    {
        int f=0x7f7f7f7f;
        for(int i=t;i!=s;i=pp[i])
            f=min(f,e[pe[i]].flow);
        F+=f;
        C+=dis[t]*f;
        for(int i=t;i!=s;i=pp[i])
        {
            e[pe[i]].flow-=f;
            e[pe[i]^1].flow+=f;
        }
    }
    return {F,C};
}

int main()
{
    scanf("%d%d%d%d",&n,&m,&s,&t);
    for(int i=1,u,v,f,c;i<=m;i++)
    {
        scanf("%d%d%d%d",&u,&v,&f,&c);
        add(u,v,f,c);
        add(v,u,0,-c);
    }
    Pair ans=work();
    printf("%d %d",ans.first,ans.second);
    return 0;
}
```



## 二分图

1. 最大匹配数=最小点覆盖
2. 最大独立集=顶点数 - 最大匹配数
3. 最小路径覆盖数=顶点数 - 原DAG图的拆点二分图的最大匹配数

### 匈牙利

```cpp
#include<vector>
#include<cstdio>
#include<cstring>

int n1,n2,m;

struct men{
	int lover;
	std::vector<int> t;
}man[1010];
int woman[1010]={0};
bool vis[1010];
int dfs(int u)
{
	for(int i=0,sz=man[u].t.size();i<sz;i++)
	{
		int v=man[u].t[i];
		if(vis[v]) continue;
		vis[v]=1;
		if(!woman[v]||dfs(woman[v]))
		{
			woman[v]=u;
			return 1;
		}
	}
	return 0;
}
int main()
{
	scanf("%d%d%d",&n1,&n2,&m);
	for(int i=1,u,v;i<=m;i++)
	{
		scanf("%d%d",&u,&v);
		man[u].t.push_back(v);
	}
	int ans=0;
	for(int i=1;i<=n1;i++)
	{
		memset(vis,0,sizeof(vis));
		ans+=dfs(i);
	}
	printf("%d\n",ans);
	return 0;
}
```

## LCA

### tarjan并查集

```cpp
#include<stdio.h>

int n,m,root;

struct Edge{
	int nxt,to;
}e[2000010];
int head[1000010]={0},cnt=1;
void add(int u,int v)
{
	e[cnt]={head[u],v};
	head[u]=cnt++;
}
struct Query{
	int nxt,to,id;
}q[2000010];
int qhead[1000010]={0},qcnt=1;
void add(int u,int v,int id)
{
	q[qcnt]={qhead[u],v,id};
	qhead[u]=qcnt++;
}

int ans[1000010]={0};

int fa[1000010]={0};
int find(int x) {return (x==fa[x])?x:(fa[x]=find(fa[x]));}
void uni(int x,int y)//x做根
{
	x=find(x);y=find(y);
	fa[y]=x;
}
bool vis[1000010]={0};
void dfs(int u,int fa)
{
	vis[u]=1;
	for(int i=head[u];i;i=e[i].nxt)
	{
		int v=e[i].to;
		if(vis[v]) continue;
		dfs(v,u);
	}
	for(int i=qhead[u];i;i=q[i].nxt)
	{
		int v=q[i].to;
		if(vis[v])
			ans[q[i].id]=find(v);
	}
	uni(fa,u);
}

int main()
{
	scanf("%d%d%d",&n,&m,&root);
	for(int i=1;i<=n;i++) fa[i]=i;
	for(int i=1,u,v;i<n;i++)
	{
		scanf("%d%d",&u,&v);
		add(u,v);
		add(v,u);
	}
	for(int i=1,u,v;i<=m;i++)
	{
		scanf("%d%d",&u,&v);
		add(u,v,i);
		add(v,u,i);
	}
	dfs(root,root);
	for(int i=1;i<=m;i++) printf("%d\n",ans[i]);
	return 0;
}
```

### 倍增

```cpp
//just a copy of code
#include<cstdio>
#include<cstring>
#include<vector>
using namespace std;
const int maxn=40005;
int t,n,m;
struct node{int v,dis;node(int v=0,int dis=0):v(v),dis(dis){}};
vector<node>edge[maxn];

int num[maxn<<1],cnt=0;
int first[maxn];
int deep[maxn];

int minn[maxn<<1][20];

int father[maxn][20];
int dis[maxn];

void dfs(int fa,int u,int dep,int ds)
{
    deep[u]=dep;
    dis[u]=ds;
    father[u][0]=fa;
    for(int i=1;(1<<i)<=dep;i++)
        father[u][i]=father[father[u][i-1]][i-1];

    num[++cnt]=u;
    first[u]=cnt;
    for(node nd:edge[u])
    {
        if(nd.v!=fa)
        {
            dfs(u,nd.v,dep+1,ds+nd.dis);
            num[++cnt]=u;
        }
    }
    return;
}

int main()
{
    scanf("%d",&t);
    while (t--)
    {
        for(int i=0;i<maxn;i++)edge[i].clear();
        scanf("%d%d",&n,&m);
        for(int i=0,u,v,ds;i<n-1;i++)
        {
            scanf("%d%d%d",&u,&v,&ds);
            edge[u].push_back(node(v,ds));
            edge[v].push_back(node(u,ds));
        }

        cnt=0;
        dfs(0,1,0,0);

        for(int j=1;j<=cnt;j++)minn[j][0]=num[j];
        for(int i=1;(1<<i)<=cnt;i++)
        {
            for(int j=1;j+(1<<i)-1<=cnt;j++)
                minn[j][i]=
                    deep[minn[j][i-1]]<deep[minn[j+(1<<(i-1))][i-1]]?
                    minn[j][i-1]:minn[j+(1<<(i-1))][i-1];
        }
        
        int a,b;
        while (m--)
        {
            scanf("%d%d",&a,&b);
            int bg=first[a],ed=first[b];
            if(bg>ed)swap(bg,ed);
            int lca=a;
            for(int i=19;i>=0 && bg<=ed;i--)
            {
                if(bg+(1<<i)-1<=ed)
                {
                    if(deep[lca]>deep[minn[bg][i]])
                        lca=minn[bg][i];
                    bg+=(1<<i);
                }
            }
            printf("%d\n",dis[a]+dis[b]-2*dis[lca]);
        }
    }
    return 0;
}
```

## 树链剖分

```cpp
#include<cstdio>
#include<algorithm>

int n,q;

struct Edge{
    int nxt,to;
}e[60010];
int head[30010],cnt=1;
void add(int u,int v)
{
    e[cnt]={head[u],v};
    head[u]=cnt++;
    e[cnt]={head[v],u};
    head[v]=cnt++;
}
struct Tree{
    long long w;
    int fa,dep,sz,wson,top,id;
}t[30010];
void dfs1(int u,int fa)
{
    t[u].fa=fa;
    t[u].dep=t[fa].dep+1;
    t[u].sz=1;
    t[u].wson=0;
    int maxn=0;
    for(int i=head[u];i;i=e[i].nxt)
    {
        int v=e[i].to;
        if(v==fa) continue;
        dfs1(v,u);
        int temp=t[v].sz;
        t[u].sz+=temp;
        if(temp>maxn)
        {
            t[u].wson=v;
            maxn=temp;
        }
    }
}
int id=1;
long long a[30010];
void dfs2(int u,int top)
{
    t[u].top=top;
    t[u].id=id;
    a[id]=t[u].w;
    id++;
    if(!t[u].wson) return;
    dfs2(t[u].wson,top);
    for(int i=head[u];i;i=e[i].nxt)
    {
        int v=e[i].to;
        if(v==t[u].fa||v==t[u].wson) continue;
        dfs2(v,v);
    }
}
struct SegTree{
    int l,r;
    long long sum,mx;
}s[120010];
inline void pushup(int x)
{
    s[x].sum=s[x<<1].sum+s[x<<1|1].sum;
    s[x].mx=std::max(s[x<<1].mx,s[x<<1|1].mx);
}
void build(int x,int l,int r)
{
    s[x].l=l;
    s[x].r=r;
    if(l==r)
    {
        s[x].mx=s[x].sum=a[l];
        return;
    }
    int mid=l+r>>1;
    build(x<<1,l,mid);
    build(x<<1|1,mid+1,r);
    pushup(x);
}
void update(int x,int pos,long long k)
{
    if(s[x].l==s[x].r&&s[x].l==pos)
    {
        s[x].mx=s[x].sum=k;
        return;
    }
    int mid=s[x].l+s[x].r>>1;
    if(pos<=mid) update(x<<1,pos,k);
    else update(x<<1|1,pos,k);
    pushup(x);
}
long long quemx(int x,int l,int r)
{
    if(l<=s[x].l&&s[x].r<=r) return s[x].mx;
    int mid=s[x].l+s[x].r>>1;
    long long ans=-1e9;
    if(l<=mid) ans=std::max(ans,quemx(x<<1,l,r));
    if(r>mid) ans=std::max(ans,quemx(x<<1|1,l,r));
    return ans;
}
long long quesum(int x,int l,int r)
{
    if(l<=s[x].l&&s[x].r<=r) return s[x].sum;
    int mid=s[x].l+s[x].r>>1;
    long long ans=0;
    if(l<=mid) ans+=quesum(x<<1,l,r);
    if(r>mid) ans+=quesum(x<<1|1,l,r);
    return ans;
}
inline void change(int pos,long long k)
{
    update(1,t[pos].id,k);
}
long long qmax(int u,int v)
{
    long long ans=-99999999;
    while(t[u].top!=t[v].top)
    {
        if(t[t[u].top].dep<t[t[v].top].dep) std::swap(u,v);
        ans=std::max(ans,quemx(1,t[t[u].top].id,t[u].id));
        u=t[t[u].top].fa;
    }
    if(t[u].id>t[v].id) std::swap(u,v);
    ans=std::max(ans,quemx(1,t[u].id,t[v].id));
    return ans;
}
long long qsum(int u,int v)
{
    long long ans=0;
    while(t[u].top!=t[v].top)
    {
        if(t[t[u].top].dep<t[t[v].top].dep) std::swap(u,v);
        ans+=quesum(1,t[t[u].top].id,t[u].id);
        u=t[t[u].top].fa;
    }
    if(t[u].id>t[v].id) std::swap(u,v);
    ans+=quesum(1,t[u].id,t[v].id);
    return ans;
}

int main()
{
    //freopen("test.in","r",stdin);
    scanf("%d",&n);
    for(int i=1,u,v;i<n;i++)
    {
        scanf("%d%d",&u,&v);
        add(u,v);
    }
    for(int i=1;i<=n;i++) scanf("%lld",&t[i].w);
    dfs1(1,0);
    dfs2(1,1);
    build(1,1,n);
    scanf("%d",&q);
    while(q--)
    {
        char opt[20]={0};
        int x,y;
        scanf("%s%d%d",opt,&x,&y);
        /*if(opt[1]=='H') change(x,(long long)y);
        else if(opt[1]=='M') printf("%lld\n",qmax(x,y));
        else printf("%lld\n",qsum(x,y));*/
    }
}
```

# 计算几何

## 凸包（含极角排序）

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

# 动态规划

## 背包

### 01背包

```cpp
for (int i = 1; i <= n; i++)
  for (int j = W; j >= w[i]; j--)
    dp[j]=max(dp[j],dp[j-w[i]]+v[i]);
```

### 完全背包

```cpp
for (int i = 1; i <= n; i++)
  for (int j = w[i]; j <= W; j++)
    dp[j]=max(dp[j],dp[j-w[i]]+v[i]);
```

### 二进制优化多重背包

```cpp
#include<bits/stdc++.h>
using namespace std;
const int maxn=2005;//maxn*=log(maxk)
const int maxm=40005;
int ww[maxn],vv[maxn],k[maxn];//不需要数组
int w[maxn],v[maxn],cnt=0;
int dp[maxm];
int main()
{
    int n,W;
    scanf("%d%d",&n,&W);
    for(int i=0;i<n;i++)
    {
        scanf("%d%d%d",&vv[i],&ww[i],&k[i]);
        //k为数量，ww[i]为原来体积,vv[i]为原来价值
        for(int tmp=1;k[i]>0;tmp<<=1)
        {
            if(k[i]>=tmp){
                w[++cnt]=tmp*ww[i];
                v[cnt]=tmp*vv[i];
            }
            else{
                w[++cnt]=k[i]*ww[i];
                v[cnt]=k[i]*vv[i];
            }
            k[i]-=tmp;
        }
    }
    for (int i = 1; i <= cnt; i++)
        for (int j = W; j >= w[i]; j--)
            dp[j]=max(dp[j],dp[j-w[i]]+v[i]);
    
    printf("%d\n",dp[W]);
    
    return 0;
}
```

## 数位dp

### 不要62

```cpp
#include<cstdio>
#include<cstring>
int dpp[10][2];//记忆化，除去jud
int num[10];//逆序
inline int dp(int digit,bool jud,bool six)//位数，是否有限制，前一位是否为6
{
    if(digit==0)return 1;
    if(!jud && dpp[digit][six]!=-1)
        return dpp[digit][six];

    int sz=jud?num[digit]:9;
    int ans=0;
    for(int i=0;i<=sz;i++)
    {
        if(six && i==2 || i==4)
            continue;
        else
            ans+=dp(digit-1,jud && i==sz,i==6);
    }
    return jud?ans:dpp[digit][six]=ans;
}
inline  int f(int x)//分解
{
    int cnt=0;
    while(x)
    {
        num[++cnt]=x%10;
        x/=10;
    }
    return dp(cnt,true,false);
}
int main()
{
    int n,m;
    memset(dpp,-1,sizeof(dpp));
    while(~scanf("%d%d",&n,&m)&&(n||m))
        printf("%d\n",f(m)-f(n-1));
    return 0;
}
```

### 0的计数

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<climits>
typedef long long ll;
ll ten[40];
int num[40];
ll dpp[40][2];
ll Numm[40];
void init()
{
    memset(dpp,-1,sizeof(dpp));
    ten[0]=1;
    for(int i=1;(double)ULLONG_MAX/ten[i-1]>=10.0;i++)
        ten[i]=ten[i-1]*10;
}
ll Num(int digit)//分解逆操作
{
    if(Numm[digit]!=-1)return Numm[digit];
    ll ans=0;
    for(int i=digit;i>=1;i--)
        ans=ans*10+num[i];
    return Numm[digit]=ans;
}
ll dp(int digit,bool jud,bool isFirst)
{
    if(!digit){
        if(isFirst)return 0;
        else return 0;
    }
    if(!jud && dpp[digit][isFirst]!=-1)
        return dpp[digit][isFirst];
    
    int sz=jud?num[digit]:9;
    ll ans=0;
    for(int i=0;i<=sz;i++)
    {
        if(i==0 && !isFirst)
        {
            if(jud && i==sz)
                ans+=Num(digit-1)+1;
            else
                ans+=ten[digit-1];
        }
        ans+=dp(digit-1,jud && i==sz,isFirst && i==0);
    }
    return jud?ans:dpp[digit][isFirst]=ans;
}
inline ll f(ll x)//分解
{
    memset(Numm,-1,sizeof(Numm));
    if(x<0)return 0;
    int cnt=0;
    Numm[0]=0;
    while(x)
    {
        num[++cnt]=x%10;
        Numm[cnt]=Numm[cnt-1]+num[cnt]*ten[cnt-1];
        x/=10;
    }
    return dp(cnt,true,true)+1;
}
int main()
{
    int t;
    ll m,n;
    init();
    scanf("%d",&t);
    for(int ca=1;ca<=t;ca++)
    {
        scanf("%lld%lld",&m,&n);
        printf("Case %d: %lld\n",ca,f(n)-f(m-1));
    }

    return 0;
}
```

## 斜率+cdq

$\Large dp[i]=max\{dp[i-1],\underset{1\le j<i}{max}\{A[i]*\frac{dp[j]*Rate[j]}{A[j]*Rate[j]+B[j]},B[i]*\frac{dp[j]}{A[j]*Rate[j]+B[j]}\}\}$

$\Large x[i]=\frac{dp[i]}{A[i]*Rate[i]+B[i]},y[i]=\frac{dp[i]*Rate[i]}{A[i]*Rate[i]+B[i]},k[i]=-\frac{B[i]}{A[i]}$

$\Large x[j]>x[k]\quad且\quad j优于k\to\frac{y[j]-y[k]}{x[j]-x[k]}>k[i]$

```cpp
#include<cstdio>
#include<cmath>
#include<cstring>
#include<algorithm>
#include<vector>
using namespace std;
const int maxn=1e5+5;
const double eps=1e-9;
const double inf=1e100;
double dp[maxn],A[maxn],B[maxn],Rate[maxn];
struct Q{int i;double x,y,k;}q[maxn],tmp[maxn];
double slope(const Q& p1,const Q&p2){
    if(abs(p2.x-p1.x)<eps)return inf;
    return (p2.y-p1.y)/(p2.x-p1.x);
}
vector<Q>convex;//队列
void cdq(int l,int r);
int main()
{
    int n;
    scanf("%d%lf",&n,&dp[0]);
    for(int i=1;i<=n;i++)
    {
        scanf("%lf%lf%lf",&A[i],&B[i],&Rate[i]);
        q[i].i=i;
        q[i].k=-B[i]/A[i];
    }
    sort(q+1,q+n+1,[](const Q&_a,const Q&_b){return _a.k>_b.k;});//按k先排序
    cdq(1,n);
    printf("%.3lf\n",dp[n]);
    return 0;
}
void cdq(int l,int r)
{
    if(l==r)//q[l].i=l=r
    {
        dp[l]=max(dp[l],dp[l-1]);
        q[l].x=dp[l]/(A[l]*Rate[l]+B[l]);
        q[l].y=dp[l]*Rate[l]/(A[l]*Rate[l]+B[l]);
        return;
    }
	//拆分
    int mid=(l+r)>>1;
    for(int i=l,p1=l,p2=mid+1;i<=r;i++)
    {
        if(q[i].i<=mid)tmp[p1++]=q[i];
        else tmp[p2++]=q[i];
    }
    memcpy(q+l,tmp+l,(r-l+1)*sizeof(Q));

    cdq(l,mid);
    //建凸壳
    convex.clear();
    for(int i=l;i<=mid;i++)
    {
        while(convex.size()>=2 && slope(convex[convex.size()-2],convex[convex.size()-1]) < slope(convex[convex.size()-1],q[i]))
            convex.pop_back();
        convex.push_back(q[i]);
    }
    //更新
    for(int i=mid+1,j=0;i<=r;i++)//j是队首
    {
        while(j<convex.size()-1 && slope(convex[j],convex[j+1]) > q[i].k)
            j++;
        dp[q[i].i]=max(dp[q[i].i],A[q[i].i]*convex[j].y+B[q[i].i]*convex[j].x);
    }

    cdq(mid+1,r);
    //按x归并排序
    {
        int i=l,j=mid+1,pt=l;
        while(i<=mid && j<=r)
        {
            if(q[i].x<q[j].x)tmp[pt++]=q[i++];
            else tmp[pt++]=q[j++];
        }
        while(i<=mid)
            tmp[pt++]=q[i++];
        while (j<=r)
            tmp[pt++]=q[j++];
        memcpy(q+l,tmp+l,(r-l+1)*sizeof(Q));       
    }
}
```

# 数学

[TOC]

##  fft

```cpp
#include<complex>
#include<cmath>
const int maxn=1<<17;//131072
const long double pi=acos(-1);
typedef std::complex<long double> cp;
cp a[maxn*2+5],b[maxn*2+5];
int rev[maxn*2+5];
void fft(cp a[],int n,int sig=1)
{
    int k=0;while(n>(1<<k))k++;
    for(int i=0;i<n;i++)rev[i]=((rev[i>>1]>>1)|((1&i)<<(k-1)));
    for(int i=0;i<n;i++)if(i<rev[i])swap(a[i],a[rev[i]]);
    for(int len=1;len<n;len<<=1)
        for(int s=0;s<n;s+=len*2)
        {
            cp w=1,delta(cos(pi/len),sig*sin(pi/len));
            for(int i=s;i<s+len;i++,w*=delta)
            {
                cp t=w*a[i+len];
                a[i+len]=a[i]-t;
                a[i]=a[i]+t;
            }
        }
    if(sig==-1)
        for(int i=0;i<n;i++)a[i]/=n;
}
int main()
{
    int n=ori_n;
    if(n!=(n&-n)){
        n<<=1;
        while(n!=(n&-n))n-=(n&-n);
    }
    fft(a,n*2);fft(b,n*2);
    a*=b;//卷积
    fft(a,n*2,-1);
    for(item:a)print((int)(item.real()+0.5))
}
```



## 线性基

```cpp
class LBase{
    bool zero=false;
    int cnt=0;
    long long data[63]={0};
    void rebuild();
public:
    bool insert(long long x);
    void clear();
    long long max();
    long long min();
    long long kth(long long k);//第k小
}lb;
bool LBase::insert(long long x)
{
    if(x<0)return false;
    for(int i=62;i>=0;i--)
    {
        if(x&(1LL<<i))
        {
            if(!this->data[i])
            {
                this->data[i]=x;
                this->cnt++;
                return true;
            }
            else
                x^=this->data[i];
        }
    }
    if(!this->zero)
    {
        this->zero=true;
        return true;
    }
    return false;
}
void LBase::clear()
{
    zero=false;
    cnt=0;
    fill(data,data+sizeof(data)/sizeof(data[0]),0);
}
long long LBase::max()
{
    long long ans=0;
    for(int i=62;i>=0;i--)
    {
        if((this->data[i]^ans)>ans)
            ans^=this->data[i];
    }
    return ans;
}
long long LBase::min()
{
    if(this->zero)return 0;
    for(int i=62;i>=0;i--)
        if(this->data[i])
            return this->data[i]; 
    return -1;
}
void LBase::rebuild()
{
    for(int i=62;i>=1;i--)
        if(this->data[i])
            for(int j=i-1;j>=0;j--)
                if(this->data[i] & (1LL<<j))
                    this->data[i]^=this->data[j];
}
long long LBase::kth(long long k)
{
    this->rebuild();
    if(k<=0)return -1;
    if(this->zero){
        k--;
        if(!k)return 0;
    }
    if(k>=(1LL<<this->cnt))return -1;
    long long ans=0;
    for(int i=0,cnt_t=0;i<=62;i++)
    {
        if(this->data[i])
        {
            if(k&(1LL<<cnt_t))
                ans^=this->data[i];
            cnt_t++;
        }
    }
    return ans;
}
```

## 求逆元

### 线性递推

```cpp
int inv[3000010];
void init()
{
    inv[1] = 1;
	for (int i = 2; i <= MAXN && i < mod; i++)
		inv[i] = 1LL*(mod - mod / i) % mod * inv[mod % i] % mod;
}
```

### 扩展欧几里得

```cpp
void exgcd(int a,int b,int &x,int &y)
{
    if(b==0){x=1;y=0;return;}
    exgcd(b,a%b,x,y);
    int t=x;x=y;y=t-a/b*y;
}
inline int inv(int t)
{
    int x,y;
    exgcd(t,mod,x,y);
    return (x%mod+mod)%mod;
}
```

### 费马小定理（附快速幂）

```cpp
int po(int x,int y)
{
	int ans=1;
	while(y)
	{
		if(y&1) ans=1LL*ans*x%mod;
		x=1LL*x*x%mod;
		y>>=1;
	}
	return ans;
}
inline int inv(int x)
{
	return po(x,mod-2);
}
```

## 筛法

### 埃氏筛

```cpp
isprime[1]=false;	
for(int i=2;i<=MAXN;i++) isprime[i]=true;
for(int i=2;i*i<=MAXN;i++)
{
	if(isprime[i])
		for(int j=2;j*i<=MAXN;j++)
			isprime[i*j]=0;
}
```

### 线性筛

```cpp
memset(isprime,1,sizeof(isprime));
isprime[1]=false;
for(int i=2;i<=MAXN;i++)
{
    if(isprime[i]) prime[++primesize]=i;
    for(int j=1;j<=primesize&&i*prime[j]<=MAXN;j++)
    {
       isprime[i*prime[j]]=false;
       if(i%prime[j]==0) break;//这里,保证是最小因子
    }
}
```

### 欧拉筛

1. n为质数时，$\varphi(n) = n - 1$
2. p为质数且p整除n时，$\varphi(n*p) = p* \varphi(n) $
3. p为质数且p不整除n时，$\varphi(n*p) =(p - 1) * \varphi(n)$

```cpp
phi[1] = 1;
for(int i = 2 ; i <= MAXN ; i++)
{
	if(!notprime[i])//是质数
	{
		prime[tot++] = i;
		phi[i] = i - 1;
	}
	for(int j = 0 ; j < tot && i * prime[j] <= MAXN ; j++)
	{
		notprime[i * prime[j]] = true;
		if(i % prime[j] == 0)
        {
			phi[i * prime[j]] = phi[i] * prime[j];
			break;
		}
		else
			phi[i * prime[j]] = phi[i] * (prime[j] - 1);
	}
}
```

## 高斯消元

```cpp
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
int n;
double m[105][105]={0};
double x[105]={0};

void debug()
{
	for(int i=1;i<=n;i++,printf("\n"))
		for(int j=1;j<=n+1;j++)
			printf("%10.5lf",m[i][j]);
	printf("\n");
}

void nosolution(int i)
{
	for(int j=1;j<=n;j++)
		if(fabs(m[i][j])>1e-5)
			return;
	printf("No Solution\n");
	exit(0);
}

int main()
{
	//freopen("test.in","r",stdin);
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
		for(int j=1;j<=n+1;j++)
			scanf("%lf",&m[i][j]);
	for(int i=1;i<=n;i++)
	{
		for(int j=i+1;j<=n;j++)
		{
			if(fabs(m[j][i])<1e-6) continue;
			double k=m[j][i]/m[i][i];
			for(int h=i;h<=n+1;h++)
			{
				m[j][h]-=m[i][h]*k;
			}
			//debug();
		}
	}
	for(int i=n;i;i--)
	{
		nosolution(i);
		x[i]=m[i][n+1]/m[i][i];
		for(int j=i-1;j;j--)
		{
			m[j][n+1]-=x[i]*m[j][i];
			m[j][i]=0;
		}
	}
	for(int i=1;i<=n;i++) printf("%.2lf\n",x[i]);
	return 0;
}
```

## 矩阵快速幂

```cpp
/*斐波那契第n项*/
#include<stdio.h>
#include<memory.h>
#define Mod 1000000007

struct m{
	long long a[2][2];
	m(){memset(a,0,sizeof(a));}
}e;//单位矩阵

m mult(m x,m y)
{
	m z;
    z.a[0][0]=(x.a[0][0]*y.a[0][0]+x.a[0][1]*y.a[1][0])%Mod;
    z.a[0][1]=(x.a[0][0]*y.a[0][1]+x.a[0][1]*y.a[1][1])%Mod;
    z.a[1][0]=(x.a[1][0]*y.a[0][0]+x.a[1][1]*y.a[1][0])%Mod;
    z.a[1][1]=(x.a[1][0]*y.a[0][1]+x.a[1][1]*y.a[1][1])%Mod;
	return z;
}
m p(m x,long long kk)
{
	m ans=e;
	while(kk)
	{
		if(kk&1) ans=mult(ans,x);
		x=mult(x,x);
		kk>>=1;
	}
	return ans;
}
int main()
{
	m mm,first;
	e.a[0][0]=e.a[1][1]=1;//e为单位矩阵
	mm.a[0][0]=mm.a[0][1]=mm.a[1][0]=1;//mm为构造矩阵
	first.a[0][0]=first.a[0][1]=1;//first为数列最开始元素
	long long n;
	scanf("%lld",&n);
	if(n==1) printf("1\n");
	else printf("%lld\n",mult(first,p(mm,n-2)).a[0][0]);
	return 0;
}

```

# 求解策略

## 贪心

1. 正难则反

   > [例1]给定拥有的1角、5角、10角、50角、100角的纸币数量和要买的东西的价钱，求购买这件物品所需的最少纸币数量和最大纸币数量
   >
   > [解1]最小纸币从大到小贪心，最大纸币把购买物价格改为(总钱数-物品价格)
   >
   > [例2]木板切割==合并石子
   > 

## 整体二分

```cpp
//区间第k大
#include<iostream>
#include<cstring>
#include<algorithm>
#include<vector>
using namespace std;
const int maxn=1e5+5;
const int maxm=1e5+5;
int a[maxn],n;

struct Q{int l,r,k,type,no;}q[maxn+maxm*2],q1[maxn+maxm*2],q2[maxn+maxm*2];//l r means number and op(insert && delete) when type==1
int qcnt;

int ans[maxn+maxm*2];

int bit[maxn];
inline int lowbit(int x){return x&-x;}
inline void insert(int x,int delta){for(;x<=n;x+=lowbit(x))bit[x]+=delta;}
inline int query(int x){int ans=0;for(;x>0;x-=lowbit(x))ans+=bit[x];return ans;}

void solve(int qb,int qe,int l,int r);

int main()
{
    ios::sync_with_stdio(false);
    int X;
 //   cin>>X;
    memset(ans,-1,sizeof(ans));
 //   while(X--)
    {
        qcnt=0;
        int m,maxa=0,mina=1e9;

        cin>>n>>m;
        for(int i=1;i<=n;i++)
        {
            cin>>a[i];
            maxa=max(maxa,a[i]);
            mina=min(mina,a[i]);
            q[qcnt]={a[i],1,i,1,qcnt};
            qcnt++;
        }
        
        char op[2];
        int i,j,k,t;
        while(m--)
        {
            cin>>op;
            if(op[0]=='Q')
            {
                cin>>i>>j>>k;
                q[qcnt]={i,j,k,2,qcnt};
                qcnt++;
            }
            else if(op[0]=='C')
            {
                cin>>i>>t;
                q[qcnt]={a[i],-1,i,1,qcnt};
                qcnt++;
                q[qcnt]={a[i]=t,1,i,1,qcnt};
                qcnt++;
                maxa=max(maxa,t);
                mina=min(mina,t);
            }
        }
        solve(0,qcnt-1,mina,maxa);
        for(int i=0;i<qcnt;i++)
            if(ans[i]!=-1)
                cout<<ans[i]<<endl,ans[i]=-1;
    }
    return 0;
}

void solve(int qb,int qe,int l,int r)
{
    if(qb>qe)return;//!!

    if(l==r)
    {
        for(int i=qb;i<=qe;i++)
            if(q[i].type==2)ans[q[i].no]=l;
        return;
    }

    int m=(l+r)>>1,cnt1=0,cnt2=0;
    for(int i=qb;i<=qe;i++)
    {
        if(q[i].type==1)
        {
            if(q[i].l<=m)
            {
                insert(q[i].k,q[i].r);
                q1[cnt1++]=q[i];
            }
            else
                q2[cnt2++]=q[i];
        }
        else if(q[i].type==2)
        {
            int cnt=query(q[i].r)-query(q[i].l-1);
            if(cnt < q[i].k)
                q[i].k-=cnt,q2[cnt2++]=q[i];
            else
                q1[cnt1++]=q[i];
        }
    }
    for(int i=qb;i<=qe;i++)
        if(q[i].type==1 && q[i].l<=m)insert(q[i].k,-q[i].r);

    memcpy(q+qb,q1,cnt1*sizeof(q[0]));
    memcpy(q+qb+cnt1,q2,cnt2*sizeof(q[0]));
    solve(qb,qb+cnt1-1,l,m);
    solve(qb+cnt1,qe,m+1,r);
}

```

# 其他

### 读取一行(不能读空白行)
```c++
char str[10];
scanf(" %[^\n]\n",str);
```

### 快读

```cpp
#define ch_top 10000000
char ch[ch_top],*now_r=ch;
void read(int &x) 
{ while(*now_r<48)++now_r;
  for (x=*now_r-48;*++now_r>=48;)
   x= (x<<1)+(x<<3)+*now_r-48;
}
int main()
{
    fread(ch,1,ch_top,stdin);
    //…………
}
```

```c++
#include <cstdio>
#include <cctype>
const int MAXSIZE = 1 << 20;
char buf[MAXSIZE], *p1, *p2;
#define gc()                                                               \
  (p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, MAXSIZE, stdin), p1 == p2) \
       ? EOF                                                               \
       : *p1++)

template <class T>
inline void read(T& x) {
    int f = 1;
    x = 0;
    char c = gc();
    while (!isdigit(c)) {
        if (c == '-') f = -1;
        c = gc();
    }
    while (isdigit(c)) x = x * 10 + (c ^ 48), c = gc();
    x *= f;
}
```

### vim

```vimrc
"配置
set nu
set nobackup
set noswapfile
set mouse=a

set tabstop=4
set softtabstop=4
set expandtab
set shiftwidth=4
set smarttab
set cindent

"快捷键
map <F5> :call Compile()<CR>
map <F2> :call SetTitle()<CR>
map <F12> :call Clear()<CR>

"函数
func! Compile()
    exec "w"
    if &filetype == 'cpp'
        exec '!g++ % -o %<'
        exec '!time ./%<'
    elseif &filetype == 'java'
        exec '!javac %'
        exec '!time java %<'
    endif
endfunc 

func SetTitle()
let l = 0
let l = l + 1 | call setline(l,'#include <cstdio>')
let l = l + 1 | call setline(l,'#include <cstring>')
let l = l + 1 | call setline(l,'#include <cmath>')
let l = l + 1 | call setline(l,'#include <cstdlib>')
let l = l + 1 | call setline(l,'#include <ctime>')
let l = l + 1 | call setline(l,'#include <algorithm>')
let l = l + 1 | call setline(l,'#include <vector>')
let l = l + 1 | call setline(l,'#include <queue>')
let l = l + 1 | call setline(l,'#include <set>')
let l = l + 1 | call setline(l,'#include <map>')
let l = l + 1 | call setline(l,'#include <string>')
let l = l + 1 | call setline(l,'using namespace std;')
let l = l + 1 | call setline(l,'')
let l = l + 1 | call setline(l,'int main()')
let l = l + 1 | call setline(l,'{')
let l = l + 1 | call setline(l,'    //freopen("in.txt","r",stdin);')
let l = l + 1 | call setline(l,'    //freopen("out.txt","w",stdout);')
let l = l + 1 | call setline(l,'    ')
let l = l + 1 | call setline(l,'    ')
let l = l + 1 | call setline(l,'    return 0;')
let l = l + 1 | call setline(l,'}')
endfunc

func Clear()
	exec '0,$d'
endfunc
```

### codeblocks

#### 编辑

1. 缩放： `ctrl + 滚轮`
2. 注释/取消注释：`ctrl+shift+C`/`ctrl+shift+X`
3. 整行(几行)移动：`alt+↑↓`
4. 关键词（if/while等）补全：`ctrl+J`

#### 编译调试

1. 编译：`ctrl+F9`
2. 编译当前文件