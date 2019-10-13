## 并查集



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

