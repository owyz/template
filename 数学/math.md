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

