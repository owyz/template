### 读取一行(不能读空白行)
```c++
char str[10];
scanf(" %[^\n]\n",str);
```

### 快读
```c++
#include <cstdio>
#include <cctype>
const int MAXSIZE = 1 << 20;
char buf[MAXSIZE], *p1, *p2;
#define gc()                                                               \
  (p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, MAXSIZE, stdin), p1 == p2) \
       ? EOF                                                               \
       : *p1++)

template <class T>
inline void read(T& x) {
    int f = 1;
    x = 0;
    char c = gc();
    while (!isdigit(c)) {
        if (c == '-') f = -1;
        c = gc();
    }
    while (isdigit(c)) x = x * 10 + (c ^ 48), c = gc();
    x *= f;
}
```

### vim

```vimrc
"配置
set nu
set nobackup
set noswapfile
set mouse=a

set tabstop=4
set softtabstop=4
set expandtab
set shiftwidth=4
set smarttab
set cindent

"快捷键
map <F5> :call Compile()<CR>
map <F2> :call SetTitle()<CR>
map <F12> :call Clear()<CR>

"函数
func! Compile()
    exec "w"
    if &filetype == 'cpp'
        exec '!g++ % -o %<'
        exec '!time ./%<'
    elseif &filetype == 'java'
        exec '!javac %'
        exec '!time java %<'
    endif
endfunc 

func SetTitle()
let l = 0
let l = l + 1 | call setline(l,'#include <cstdio>')
let l = l + 1 | call setline(l,'#include <cstring>')
let l = l + 1 | call setline(l,'#include <cmath>')
let l = l + 1 | call setline(l,'#include <cstdlib>')
let l = l + 1 | call setline(l,'#include <ctime>')
let l = l + 1 | call setline(l,'#include <algorithm>')
let l = l + 1 | call setline(l,'#include <vector>')
let l = l + 1 | call setline(l,'#include <queue>')
let l = l + 1 | call setline(l,'#include <set>')
let l = l + 1 | call setline(l,'#include <map>')
let l = l + 1 | call setline(l,'#include <string>')
let l = l + 1 | call setline(l,'using namespace std;')
let l = l + 1 | call setline(l,'')
let l = l + 1 | call setline(l,'int main()')
let l = l + 1 | call setline(l,'{')
let l = l + 1 | call setline(l,'    //freopen("in.txt","r",stdin);')
let l = l + 1 | call setline(l,'    //freopen("out.txt","w",stdout);')
let l = l + 1 | call setline(l,'    ')
let l = l + 1 | call setline(l,'    ')
let l = l + 1 | call setline(l,'    return 0;')
let l = l + 1 | call setline(l,'}')
endfunc

func Clear()
	exec '0,$d'
endfunc
```