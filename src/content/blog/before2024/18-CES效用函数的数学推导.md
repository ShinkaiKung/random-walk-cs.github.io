---
author: 公子小白
pubDatetime: 2020-06-01T19:56:00Z
title: CES效用函数的数学推导
postSlug: "18"
featured: false
draft: false
tags:
  - 经济学
description: CES效用函数的数学推导
---

**CES效用函数：** $U(x,y)=\frac{x^\delta}{\delta}+\frac{y^\delta}{\delta},\delta\le 1$

**当 $\delta=1$ 时，得到完全替代效用函数：**

$U(x,y)=x+y$ .

**当 $\delta=0$ 时，得到柯布道格拉斯效用函数：**

单调变换得： $U^*(x,y)=\frac{x^\delta-1}{\delta}+\frac{y^\delta-1}{\delta}=\frac{x^\delta+y^\delta-2}{\delta}$ .

$\frac{0}{0}$ 型： $\lim\limits_{\delta\to 0}U^*(x,y)=\lim\limits_{\delta\to 0}\frac{x^\delta lnx+y^\delta lny}{1}=lnx+lny$ .

**当 $\delta=-\infty$ 时，得到完全互补效用函数：**(经单调变换， $x>0,y>0$ 可化为 $x>1,y>1$ )

单调变换得： $U^*(x,y)=\frac{ln(x^\delta+y^\delta)}{\delta}$ .

$\lim\limits_{\delta\to-\infty}U^*(x,y)=\lim\limits_{\delta\to-\infty}\frac{ln(x^\delta+y^\delta)}{\delta}$ .

$\frac{\infty}{\infty}$ 型： $\lim\limits_{\delta\to-\infty}U^*(x,y)=\lim\limits_{\delta\to-\infty}\frac{ln(x^\delta+y^\delta)}{\delta}=\lim\limits_{\delta\to-\infty}\frac{x^\delta lnx+y^\delta lny}{x^\delta+y^\delta}$ .

当 $x>y$ 时， $U^*(x,y)=\lim\limits_{\delta\to-\infty}\frac{(x/y)^\delta lnx+lny}{(x/y)^\delta+1}=lny$ .

当 $x< y$ 时， $U^*(x,y)=\lim\limits_{\delta\to-\infty}\frac{lnx+(y/x)^\delta}{1+(y/x)^\delta}=lnx$ .

当 $x=y$ 时， $U^*(x,y)=lnx=lny$ .

因此， $U^*(x,y)=min\{lnx,lny\}=min\{x,y\}$ .（单调变换）
