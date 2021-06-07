C     ******************************************************************
C     ******************************************************************

      subroutine algencan(epsfeas,epsopt,efacc,eoacc,iprint,ncomp,n,x,l,
     +u,m,lambda,equatn,linear,coded,checkder,fu,cnormu,snorm,nlpsupn,
     +inform,ifile)

      implicit none

C     SCALAR ARGUMENTS
      logical checkder
      integer inform,iprint,m,n,ncomp
      double precision cnormu,efacc,eoacc,epsfeas,epsopt,fu,nlpsupn,
     +        snorm
      
      character*(*) ifile
      
C     ARRAY ARGUMENTS
      logical coded(11),equatn(m),linear(m)
      double precision l(n),lambda(m),u(n),x(n)

      include "dim.par"
      include "machconst.inc"
      include "algconst.par"
      include "counters.inc"
      include "outtyp.inc"
      include "algparam.inc"
      include "scaling.inc"
      include "slacks.inc"
      include "fixvar.inc"

C     PARAMETERS
      integer n0,n1,n2
      parameter ( n0 =   500 )
      parameter ( n1 = 10000 )
      parameter ( n2 = 20000 )

C     LOCAL SCALARS
      logical innfail,lss,scl
      integer alinfo,fcsubt,geninfo,i,iter,j,maxit,nwcalls,nwtotit,
     +        oprint,outiter,solinfo,totiter,msqcalls,msqtotit,toteccnt,
     +        totejccnt,totehccnt,aveeccnt,aveejccnt,aveehccnt
      double precision cnorm,cnormb,cnormub,f,fb,fub,ncsupn,nlpsupnb,
     +        rsupn
      real time

C     LOCAL ARRAYS
      character * 4 lsssub,sclsub
      double precision nl(nmax),c(mmax),cu(mmax),rho(mmax)
      real dum(2)

C     DATA STATEMENTS
      data dum/0.0,0.0/

C     EXTERNAL FUNCTIONS AND SUBROUTINES
      external fparam,checkd,auglag,gencan,lss,scl
      external evalf,evalg,evalh,evalc,evaljac,evalhc,evalfc,evalgjac,
     +        evalgjacp,evalhl,evalhlp

C     ==================================================================
C     Start timing
C     ==================================================================

      time = dtime(dum)

C     ==================================================================
C     Set safety mode
C     ==================================================================

      safemode = .false.

C     ==================================================================
C     Initialization
C     ==================================================================

C     Set machine-dependent constants

      bignum    = 1.0d+99
      macheps   = 1.0d-16
      macheps12 = sqrt( macheps )
      macheps13 = macheps ** ( 1.0d0 / 3.0d0 )
      macheps23 = macheps ** ( 2.0d0 / 3.0d0 )

C     Set global counters

      fcnt     = 0
      efcnt    = 0
      efccnt   = 0
      egcnt    = 0
      egjccnt  = 0
      egjcpcnt = 0
      ehcnt    = 0
      ehlcnt   = 0
      ehlpcnt  = 0

      do j = 1,m
          eccnt(j)  = 0
          ejccnt(j) = 0
          ehccnt(j) = 0
      end do

C     ==================================================================
C     Check user provided subroutines
C     ==================================================================

C     Set user-provided subroutines indicators

      fcoded     = coded(1)
      gcoded     = coded(2)
      hcoded     = coded(3)
      ccoded     = coded(4)
      jaccoded   = coded(5)
      hccoded    = coded(6)
      fccoded    = coded(7)
      gjaccoded  = coded(8)
      gjacpcoded = coded(9)
      hlcoded    = coded(10)
      hlpcoded   = coded(11)

C     Check whether mandatory subroutines are being properly provided.

C     For unconstrained and bound-constrained problems, EVALF must be 
C     coded by the user. For constrained problems, EVALF and EVALC, or, 
C     alternatively, EVALFC must be coded. (Note that EVALF/EVALC and 
C     EVALFC should not be provided concurrently.) For feasibility 
C     problems, a constant null objective function must be coded and the 
C     problem solved with the IGNORE-OBJECTIVE-FUNCTION keyword. Coded 
C     subroutines must be indicated by setting the entrances of array 
C     named CODED within subroutine INIP.

C     Moreover, to avoid odd combinations, only the following choices
C     will be considered valid:
C
C     If the objective function and the constraints are given by evalf 
C     and evalc, respectively, first derivatives must be given by evalg 
C     and evaljac, while second derivatives must be given by evalh and 
C     evalhc.
C
C     If the objective function and the constraints are given by evalfc
C     then first derivatives may be given by evalgjac or evalgjacp. If
C     first derivatives are given by evalgjac, second derivatives must 
C     be given by evalhl. On the other hand, if first derivatives are
C     given by evalgjacp, second derivatives must be given by evalhlp.
C
C     Any other odd combination will be ignored.
C
C     Variables being set below have the following meaning:
C
C     firstde: whether, following the rules dictated above, the user-
C     provided subroutines will allow the method to compute first 
C     derivatives. 
C
C     seconde: whether, following the rules dictated above, the user-
C     provided subroutines will allow the method to compute the Hessian 
C     matrix of the augmented Lagrangian (to minimize the augmented 
C     Lagrangian subproblems using a second-order method) and/or the
C     Hessian of the Lagrangian plus the Jacobian of the constraints
C     (in order to solve the KKT system by Newton's method). 
C
C     truehpr: whether, following the rules dictated above, the user-
C     provided subroutines will allow the method to compute the 
C     product of the true Hessian of the augmented Lagrangian times
C     a given vector. It would allow the method to solve the augmented
C     Lagrangian subproblems using a truncated-Newton method using
C     Conjugate Gradients to solve the Newtonian linear systems.

      fcsubt = 0      
      if ( fcoded .and. ( ccoded .or. m .eq. 0 ) ) then
          fcsubt = 1
          firstde = .false.
          seconde = .false.
          truehpr = .false.
          if ( gcoded .and. ( jaccoded .or. m .eq. 0 ) ) then
              firstde = .true.
              if ( hcoded .and. ( hccoded .or. m .eq. 0 ) ) then
                  seconde = .true.
                  truehpr = .true.
              end if
          end if
      else if ( fccoded ) then
          fcsubt = 2
          firstde = .false.
          seconde = .false.
          truehpr = .false.
          if ( gjaccoded ) then
              firstde = .true.
              if ( hlcoded ) then
                  seconde = .true.
                  truehpr = .true.
              end if
          else if ( gjacpcoded ) then
              firstde = .true.
              if ( hlpcoded ) then
                  truehpr = .true.
              end if
          end if 
      end if

C     ==================================================================
C     Set default values for algoritmic parameters
C     ==================================================================

C     Hessian-vector product strategy: HAPPRO, INCQUO or TRUEHP. TRUEHP
C     is the default option. If the proper subroutines were not coded by
C     the user, then HAPPRO is used instead. HAPPRO reduces to INCQUO in
C     the unconstrained and bound-constrained cases and switches between
C     INCQUO and the product by an approximation of the Hessian in the
C     constrained case.

      if ( truehpr ) then
          hptype = 'TRUEHP'
      else
          hptype = 'HAPPRO'
      end if

C     Inner solver method: TR (trust regions, i.e. BETRA), NW (Newtonian
C     system with the use of a direct solver, i.e. MA57AD), TN + TRUEHP
C     (Truncated Newton and true Hessian-vector product) or TN + HAPPRO
C     (Truncated Newton and product with Hessian approximation). If the
C     proper subroutines were not provided by the user then TN + HAPPRO
C     is used.

      if ( seconde .and. lss(lsssub) .and. n .le. n0 ) then
          innslvr = 'TR'
      else if ( seconde .and. lss(lsssub) .and. n .le. n1 ) then
          innslvr = 'NW'
      else
          innslvr = 'TN'
          if ( n .gt. n2 ) then
              hptype  = 'HAPPRO'
          end if
      end if

C     Scaling of linear systems

      if ( lsssub .eq. 'MA57' .or. scl(sclsub) ) then
          sclsys = .true.
      else
          sclsys = .false.
      end if

C     Ignore objective function (to only find a feasible point by
C     minimizing 1/2 of the squared infeasibility)
      ignoref = .false.

C     Acceleration step
      if ( seconde .and. lss(lsssub) ) then
          skipacc = .false.
      else
          skipacc = .true.
      end if

C     Slacks for inequality constraints
      slacks = .false.

C     Remove fixed variables (with identical lower and upper bounds)
      rmfixv = .true.

C     Scale objective function and constraints
      if ( m .gt. 0 .and. .not. ignoref ) then
          scale = .true.
      else
          scale = .false.
      end if

C     Flags used to make inner calls to Gencan
      innercall = .false.
      useustp   = .false.

C     ==================================================================
C     Main output control (silent-mode?)
C     ==================================================================

      iprintctl(1) = .true.  ! Banner
      iprintctl(2) = .true.  ! Parameters and problem processing
      iprintctl(3) = .true.  ! Warnings and errors messages
      iprintctl(4) = .true.  ! Screen-mirror file algencan.out
      iprintctl(5) = .true.  ! Solution file solution.txt
      iprintctl(6) = .false. ! Statistics file with table line
      iprintctl(7) = .true.  ! User-provided subs calls counters and timing
 
      oprint = iprint

      if (oprint .eq. 0 ) then
          do i = 1,7
              iprintctl(i) = .false.
          end do
      end if

      if ( iprintctl(4) ) then
          open(unit=10,file=ifile,status='replace')
      else
          open(unit=10,           status='scratch')
      end if

C     ==================================================================
C     Set solver arguments using the specification file
C     ==================================================================

      call fparam(epsfeas,epsopt,efacc,eoacc,oprint,ncomp)

C     Outer and inner iterations output detail

      iprintout = oprint / 10
      iprintinn = mod( oprint, 10 )

C     Error tracker

      inform = 0

C     Stop if mandatory subroutines were not properly provided

      if ( fcsubt .eq. 0 ) then
          if ( iprintctl(3) ) then
C              write(* ,1000)
              write(10,1000)
          end if

          if ( safemode ) then
              inform = - 88
              call reperr(inform)
              return
          end if
      end if

C     ==================================================================
C     Initialize problem data structures
C     ==================================================================

      call sinip(n,x,l,u,m,lambda,equatn,linear,coded,checkder,inform)
      if ( inform .lt. 0 ) return

      nprint = min( n, ncomp )
      mprint = min( m, ncomp )

C     ==================================================================
C     Call the solver
C     ==================================================================

C     ALGENCAN for PNL problems

      if ( .not. ignoref .and. m .gt. 0 ) then
          call auglag(n,x,l,u,m,lambda,equatn,linear,epsfeas,epsopt,
     +    efacc,eoacc,f,c,cnorm,snorm,nl,nlpsupn,fu,cu,cnormu,fub,
     +    cnormub,fb,cnormb,nlpsupnb,ncsupn,rsupn,outiter,totiter,
     +    nwcalls,nwtotit,msqcalls,msqtotit,innfail,alinfo,inform)

          solinfo = alinfo

C     GENCAN for box-constrained problems and feasibility problems

      else
          maxit = 999999999

C         Used in feasibility problems (ignoref=true). With lambda=0 and
C         rho=1, to minimize 1/2 of the squared infeasibility coincides
C         with minimizing the augmented Lagrangian.
          do j = 1,m
              lambda(j) = 0.0d0
              rho(j)    = 1.0d0
          end do

          call gencan(n,x,l,u,m,lambda,equatn,linear,rho,epsfeas,epsopt,
     +    maxit,iter,f,nl,nlpsupn,cnorm,cnormu,geninfo,inform)

          solinfo  = geninfo

          ncsupn   = 0.0d0
          rsupn    = 0.0d0

          outiter  = 0
          totiter  = iter
          nwcalls  = 0
          nwtotit  = 0
          msqcalls = 0
          msqtotit = 0
          innfail  = .false.

          if ( ignoref ) then
              f = 0.0d0
          end if
          fu = f

          fb       = f
          fub      = fu
          cnormb   = cnorm
          cnormub  = cnormu
          nlpsupnb = nlpsupn
      end if

      if ( inform .lt. 0 ) return

C     ==================================================================
C     End problem data structures
C     ==================================================================

      call sendp(n,x,l,u,m,lambda,equatn,linear,inform)
      if ( inform .lt. 0 ) return

C     ==================================================================
C     Stop timing
C     ==================================================================

      time = dtime(dum)
      time = dum(1)

C     ==================================================================
C     Write statistics
C     ==================================================================

      if ( iprintctl(7) ) then
C          write(* ,9000) time
          write(10,9000) time

          toteccnt  = 0
          totejccnt = 0
          totehccnt = 0

          do j = 1,m
              toteccnt  = toteccnt  + eccnt(j)
              totejccnt = totejccnt + ejccnt(j)
              totehccnt = totehccnt + ehccnt(j)
          end do

          if ( m .gt. 0 ) then
              aveeccnt  = toteccnt  / m
              aveejccnt = totejccnt / m
              aveehccnt = totehccnt / m
          else
              aveeccnt  = 0
              aveejccnt = 0
              aveehccnt = 0
          end if

C          write(* ,9010) fcoded,efcnt,gcoded,egcnt,hcoded,ehcnt,ccoded,
C     +                   toteccnt,aveeccnt,jaccoded,totejccnt,aveejccnt,
C     +                   hccoded,totehccnt,aveehccnt,fccoded,efccnt,
C     +                   gjaccoded,egjccnt,gjacpcoded,egjcpcnt,hlcoded,
C     +                   ehlcnt,hlpcoded,ehlpcnt
          write(10,9010) fcoded,efcnt,gcoded,egcnt,hcoded,ehcnt,ccoded,
     +                   toteccnt,aveeccnt,jaccoded,totejccnt,aveejccnt,
     +                   hccoded,totehccnt,aveehccnt,fccoded,efccnt,
     +                   gjaccoded,egjccnt,gjacpcoded,egjcpcnt,hlcoded,
     +                   ehlcnt,hlpcoded,ehlpcnt
      end if

C     ==================================================================
C     Close output file
C     ==================================================================

      close(10)

C     ==================================================================
C     Write statistics file with table line
C     ==================================================================

      if ( iprintctl(6) ) then
          open(20,file='algencan-tabline.out')
          write(20,9040) fu,cnormu,f,cnorm,nlpsupn,fub,cnormub,fb,
     +                   cnormb,nlpsupnb,ncsupn,rsupn,inform,solinfo,
     +                   innfail,n,m,outiter,totiter,fcnt,nwcalls,
     +                   nwtotit,msqcalls,msqtotit,time
          close(20)
      end if

C     ==================================================================
C     NON-EXECUTABLE STATEMENTS
C     ==================================================================

 1000 format(/,1X,'*** Mandatory subroutines are not being ',
     +            'provided properly ***',/,1X,'For unconstrained and ',
     +            'bound-constrained problems, EVALF must be coded by ',
     +            'the',/,1X,'user. For constrained problems, EVALF ',
     +            'and EVALC, or, alternatively, EVALFC',/,1X,'must ',
     +            'be coded. (Note that EVALF/EVALC and EVALFC should ',
     +            'not be provided',/,1X,'concurrently.) For ',
     +            'feasibility problems, a constant ',
     +            'null objective function',/,1X,'must be coded and ',
     +            'the problem solved with the ',
     +            'IGNORE-OBJECTIVE-FUNCTION',/,1X,'keyword. Coded ',
     +            'subroutines must be indicated by setting ',
     +            'the entrances of array',/,1X,'named CODED ',
     +            'within subroutine INIP.')
 9000 format(/,1X,'Total CPU time in seconds: ',F8.2)
 9010 format(/,1X,'User-provided subroutines calls counters: ',/,
     +       /,1X,'Subroutine evalf     (coded=',L1,'): ',I6,
     +       /,1X,'Subroutine evalg     (coded=',L1,'): ',I6,
     +       /,1X,'Subroutine evalh     (coded=',L1,'): ',I6,
     +       /,1X,'Subroutine evalc     (coded=',L1,'): ',I6,
     +         1X,'(',I6,' calls per constraint in average)',
     +       /,1X,'Subroutine evaljac   (coded=',L1,'): ',I6,
     +         1X,'(',I6,' calls per constraint in average)',
     +       /,1X,'Subroutine evalhc    (coded=',L1,'): ',I6,
     +         1X,'(',I6,' calls per constraint in average)',
     +       /,1X,'Subroutine evalfc    (coded=',L1,'): ',I6,
     +       /,1X,'Subroutine evalgjac  (coded=',L1,'): ',I6,
     +       /,1X,'Subroutine evalgjacp (coded=',L1,'): ',I6,
     +       /,1X,'Subroutine evalhl    (coded=',L1,'): ',I6,
     +       /,1X,'Subroutine evalhlp   (coded=',L1,'): ',I6,/)
 9040 format(1X,1P,D24.16,1X,1P,D7.1,1X,1P,D24.16,1X,1P,D7.1,1X,1P,D7.1,
     +       1X,1P,D24.16,1X,1P,D7.1,1X,1P,D24.16,1X,1P,D7.1,1X,1P,D7.1,
     +       1X,1P,D7.1,1X,1P,D7.1,1X,I3,1X,I1,1X,L1,1X,I6,1X,I6,1X,I3,
     +       1X,I7,1X,I7,1X,I2,1X,I7,1X,I7,1X,I7,0P,F8.2)

      end
