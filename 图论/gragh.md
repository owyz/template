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
long long dis[maxn][maxn];
memset(dis,0x3f,sizeof(dis));
for(int i=1;i<=n;i++)dis[i][i]=0;
void floyd(int n)
{
    for(int k=1;k<=n;k++)
        for(int i=1;i<=n;i++)
            for(int j=1;j<=n;j++)
                dis[i][j]=min(dis[i][j],dis[i][k]+dis[k][j]);
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
            if(!maxflow)break;//没什么软用(和优化1等效?)
        }
    }
    if(!ans)deep[u]=-1;//优化2
    return ans;
}
```

### 费用流

```cpp
//1.求最短路
//2.对s-t这一条最短路径进行增广
int dis[MAXN];
int pre[MAXN];
bool vis[MAXN];
queue<int>q;
int sumvalue;

inline void spfa()
{
    for(int i=0;i<sizeof(dis)/sizeof(int);i++)dis[i]=-INF;//dis
    memset(vis,false,sizeof(vis));//vis
    memset(pre,-1,sizeof(pre));//pre
    for(dis[s]=0,vis[s]=true,q.push(s); !q.empty(); q.pop())
    {
        int u=q.front();
        for(int i=head[u];i!=-1;i=edge[i].nex)
        {
            if(edge[i].flow>0 && dis[edge[i].v]<dis[u]+edge[i].value)
            {
                dis[edge[i].v]=dis[u]+edge[i].value;
                pre[edge[i].v]=i;
                if(!vis[edge[i].v])
                {
                    q.push(edge[i].v);
                    vis[edge[i].v]=true;
                }
            }
        }
        vis[u]=false;
    }
}

inline void flow()
{
    int flow=INF;
    for(int i=pre[t];i!=-1 && edge[i].v!=s;i=pre[edge[i^1].v])//最短路径
        flow=min(flow,edge[i].flow);

    for(int i=pre[t];i!=-1 && edge[i].v!=s;i=pre[edge[i^1].v])//修改残余网络
    {
        edge[i].flow-=flow;
        edge[i^1].flow+=flow;
        sumvalue+=edge[i].value*flow;//修改答案
    }
}

int main()
{
    for(sumvalue=0;;){
        spfa();
        if(dis[t]<=0)break;
        flow();
    }
}
```



## 二分图

1. 最大匹配数=最小点覆盖
2. 最大独立集=顶点数 - 最大匹配数
3. 最小路径覆盖数=顶点数 - 原DAG图的拆点二分图的最大匹配数

## LCA

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

