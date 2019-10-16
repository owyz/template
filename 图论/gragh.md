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



# 树

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





