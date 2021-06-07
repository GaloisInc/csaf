      subroutine ksprnt (ipflag,iprnt1,iprnt2,x,obj,g,df,dg,side,scale,
     1                   nodim,ncdim,temp,work)
      implicit double precision (a-h,o-z)
      common /kscomm/ rdf   ,adf   ,fdl   ,fdm   ,rho   ,drho  ,rhomax,
     1                fun0  ,slope ,delx  ,alpha ,alpmax,a1    ,a2    ,
     2                a3    ,a4    ,f1    ,f2    ,f3    ,f4    ,alim  ,
     3                atest ,ftest ,ifscl ,ifoff ,isx   ,isx0  ,isxlb ,
     4                isxub ,iscl  ,ig0   ,idf   ,islp  ,iobj0 ,iy    ,
     5                ip    ,ih    ,ihess ,iside ,isact ,idobj ,idg   ,
     6                itmp1 ,itmp2 ,inext ,jnext ,jsel  ,itcnt ,icntr ,
     7                icnta ,isdflg,isdrst,ifncl ,nunit ,ndv   ,ncon  ,
     8                nobj  ,nside ,nscale,iprnt ,itmax ,igrad ,limit
      character*4 ip1
      dimension x(*),obj(*),g(*),df(nodim,*),dg(ncdim,*)
      dimension side(*),scale(*),temp(*),work(*)
      data ip1 /'(   '/
c
c          main print routine
c            if (ipflag = 1) print iteration data
c            if (ipflag = 2) print final optimization data
c
c          author   - Gregory A. Wrenn
c          location - Lockheed Engineering and Sciences Co.
c                     144 Research Drive
c                     Hampton, Va. 23666
c
c          last modification - 19 July 1996
c
      if (ipflag .ne. 1) go to 200
c
c          iteration data
c
  100 continue
      if (iprnt1 .eq. 0) go to 300
      if (iprnt1 .lt. 2 .and. itcnt .gt. 1) go to 300
c
      call ksunsc (x,work(isx),work(iscl),ndv)
      write (nunit,1100) itcnt
      write (nunit,1110) rho
c
c          design variables
c
      write (nunit,1120)
      write (nunit,1000) (ip1,i,x(i),i=1,ndv)
c
c          scaling vector
c
      if (nscale .eq. 0) go to 110
      write (nunit,1130)
      write (nunit,1000) (ip1,i,scale(i),i=1,ndv)
  110 continue
c
c          objective functions and their gradients
c
      do 120 j = 1,nobj
        write (nunit,1140) j,obj(j)
        if (iprnt2 .eq. 0) go to 120
        write (nunit,1150)
        write (nunit,1000) (ip1,i,(df(j,i) / scale(i)),i=1,ndv)
  120 continue
c
c          constraints and their gradients
c
      if (ncon .eq. 0) go to 140
      write (nunit,1160)
      write (nunit,1000) (ip1,i,g(i),i=1,ncon)
      if (iprnt2 .eq. 0) go to 140
      do 130 j = 1,ncon
        write (nunit,1170) j
        write (nunit,1000) (ip1,i,(dg(j,i) / scale(i)),i=1,ndv)
  130 continue
  140 continue
c
c          side constraints
c
      if (nside .eq. 0) go to 160
      m = 0
      do 150 i = 1,ndv
        if (side(i) .eq. 0.0) go to 150
        m = m + 1
        if (side(i) .eq.  -1.0) temp(m) = float(-i)
        if (side(i) .eq.   1.0) temp(m) = float(i)
        if (side(i) .ne. 999.0) go to 150
        temp(m) = float(-i)
        m = m + 1
        temp(m) = float(i)
  150 continue
      if (m .eq. 0) go to 160
      write (nunit,1180)
      write (nunit,1190) (int(temp(i)),i=1,m)
  160 continue
c
c          print search direction and slope
c
      if (iprnt1 .lt. 3) go to 300
      write (nunit,1340) slope
      write (nunit,1000) (ip1,i,work(islp+i-1),i=1,ndv)
c
c          print hessian matrix
c
      if (iprnt1 .lt. 4) go to 300
      if (isdflg .eq. 0) write (nunit,1350)
      k = 1
      write (nunit,1360)
      do 170 i = 1,ndv
        l = k + i - 1
        write (nunit,1370) i,(work(ih+j-1),j=k,l)
        k = l + 1
  170 continue
      go to 300
c
c          final optimization data
c
  200 continue
      if (ipflag .ne. 2) go to 300
      if (iprnt1 .eq. 0) go to 300
c
      call ksunsc (x,work(isx),work(iscl),ndv)
      write (nunit,1200)
      write (nunit,1210)
      write (nunit,1000) (ip1,i,x(i),i=1,ndv)
      write (nunit,1220) rho
      do 210 j = 1,nobj
        write (nunit,1230) j,obj(j)
  210 continue
      if (ncon .eq. 0) go to 220
      write (nunit,1240)
      write (nunit,1000) (ip1,i,g(i),i=1,ncon)
  220 continue
      if (nside .eq. 0) go to 240
      m = 0
      do 230 i = 1,ndv
        if (side(i) .eq. 0.0) go to 230
        m = m + 1
        if (side(i) .eq.  -1.0) temp(m) = float(-i)
        if (side(i) .eq.   1.0) temp(m) = float(i)
        if (side(i) .ne. 999.0) go to 230
        temp(m) = float(-i)
        m = m + 1
        temp(m) = float(i)
  230 continue
      if (m .eq. 0) go to 240
      write (nunit,1250)
      write (nunit,1260) (int(temp(i)),i=1,m)
  240 continue
      write (nunit,1270)
      write (nunit,1280) itcnt
      write (nunit,1290) ifncl
      write (nunit,1300)
      if (itcnt .ge. itmax) write (nunit,1310)
      if (icntr .ge. 3    ) write (nunit,1320)
      if (icnta .ge. 3    ) write (nunit,1330)
c
  300 continue
      return
c
 1000 format (3(3x,a1,i5,2h)=,e14.7))
 1100 format (/33(2h##)/30h       Beginning of iteration ,i4)
 1110 format (34h       current value of rho is    ,e12.5)
 1120 format (/23h       design variables)
 1130 format (/21h       scaling vector)
 1140 format (/33h       objective function number ,i3,3h = ,e24.16)
 1150 format (/38h       gradients of objective function)
 1160 format (/24h       constraint values)
 1170 format (/32h       gradients for constraint ,i4)
 1180 format (/29h       upper and lower bounds)
 1190 format (5x,15i4)
c
 1200 format (37h1      final optimization information)
 1210 format (/29h       final design variables)
 1220 format (/32h       final value of rho is    ,e12.5)
 1230 format (/39h       final objective function number ,i3,3h = ,
     1        e24.16)
 1240 format (/30h       final constraint values)
 1250 format (/35h       final upper and lower bounds)
 1260 format (5x,15i4)
 1270 format (/30h       termination information)
 1280 format (/28h       number of iterations=,i6)
 1290 format (/42h       objective functions were evaluated ,i4,
     1        6h times)
 1300 format (/34h       optimization terminated by )
 1310 format (/37h       iteration count exceeded itmax)
 1320 format (/44h       relative change less than rdfun for 3,
     1        11h iterations)
 1330 format (/44h       absolute change less than adfun for 3,
     1        11h iterations)
 1340 format (/28h       slope d(ks)/d(alpha)=,e24.16/
     1        /23h       search direction)
 1350 format (/27h       hessian matrix reset)
 1360 format (/21h       hessian matrix/)
 1370 format (11h       row ,i3,3x,6e10.3/(17x,6e10.3/))
c
      end
