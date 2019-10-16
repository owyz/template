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

