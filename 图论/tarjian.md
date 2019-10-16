## tarjian

```cpp
stack<int,vector<int> >stk;
int dfn[maxn],low[maxn],instk[maxn],dfs;
int block[maxn],blk;//i点所属的块

for(int i=1;i<=n;i++)
    if(!dfn[i])tarjan(i);
void tarjan(int u)
{
    dfn[u]=low[u]=++dfs;
    stk.push(u);
    instk[u]=1;
    for(int i=head[u],v;i!=-1;i=edge[i].next)
    {
        v=edge[i].v;
        if(!dfn[v])
        {
            tarjan(v);
            low[u]=min(low[u],low[v]);
        }
        else if(instk[v])
            low[u]=min(low[u],dfn[v]);
    }
    
    if(dfn[u]==low[u])
    {
        block[u]=++blk;
        instk[u]=0;
        while(stk.top()!=u)
        {
            block[stk.top()]=block[u];
            instk[stk.top()]=0;
            stk.pop();
        }
        stk.pop();//pop u
    }
}
```

