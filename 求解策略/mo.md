## 莫队
```cpp
int sz=sqrt(n);
bool cmp(const node& a,const node& b){
    if(a.l/sz==b.l/sz)
        return a.r<b.r;
    return a.l/sz<b.l/sz;
}
for(int i=0,l=1,r=0;i<v.size();i++){
    	while(l>v[i].l)add(--l);
        while(r<v[i].r)add(++r);
        while(l<v[i].l)del(l++);
        while(r>v[i].r)del(r--);
        //get new ans
}
```

