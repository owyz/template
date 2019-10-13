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