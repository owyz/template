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

