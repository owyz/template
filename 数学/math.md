## fft

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