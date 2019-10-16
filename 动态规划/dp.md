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

