C     ******************************************************************
C     ******************************************************************
      subroutine fparam(epsfeas,epsopt,efacc,eoacc,iprint,ncomp)

      implicit none

C     SCALAR ARGUMENTS
      integer ncomp,iprint
      double precision efacc,eoacc,epsfeas,epsopt

      include "dim.par"
      include "algparam.inc"
      include "outtyp.inc"
      include "fixvar.inc"
      include "slacks.inc"
      include "scaling.inc"

C     PARAMETERS
      integer nwords
      parameter ( nwords = 18 )

C     LOCAL SCALARS
      logical lss,scl
      integer i,ifirst,ikey,ilast,inum,j
      double precision dnum

C     LOCAL ARRAYS
      character * 80 line
      character * 10 keyword
      character *  4 lsssub,lssdes,sclsub,scldes

C     DATA BLOCKS
      character * 1  addinfo(nwords),lower(26),upper(26)
      character * 10 dictionary(nwords)
      character * 41 description(nwords)

      data lower /'a','b','c','d','e','f','g','h','i','j','k','l','m',
     +            'n','o','p','q','r','s','t','u','v','w','x','y','z'/
      data upper /'A','B','C','D','E','F','G','H','I','J','K','L','M',
     +            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'/

      data dictionary( 1) /'INCREMENTA'/
      data dictionary( 2) /'HESSIAN-AP'/
      data dictionary( 3) /'TRUE-HESSI'/
      data dictionary( 4) /'SKIP-ACCEL'/
      data dictionary( 5) /'TRUST-REGI'/
      data dictionary( 6) /'NEWTON-LIN'/
      data dictionary( 7) /'TRUNCATED-'/
      data dictionary( 8) /'LINEAR-SYS'/
      data dictionary( 9) /'FIXED-VARI'/
      data dictionary(10) /'ADD-SLACKS'/
      data dictionary(11) /'OBJECTIVE-'/
      data dictionary(12) /'IGNORE-OBJ'/
      data dictionary(13) /'FEASIBILIT'/
      data dictionary(14) /'OPTIMALITY'/
      data dictionary(15) /'ACC-FEASIB'/
      data dictionary(16) /'ACC-OPTIMA'/
      data dictionary(17) /'OUTPUT-DET'/
      data dictionary(18) /'NCOMP-ARRA'/

      data description( 1) /'INCREMENTAL-QUOTIENTS-IN-CG              '/
      data description( 2) /'HESSIAN-APPROXIMATION-IN-CG              '/
      data description( 3) /'TRUE-HESSIAN-PRODUCT-IN-CG               '/
      data description( 4) /'SKIP-ACCELERATION-STEP                   '/
      data description( 5) /'TRUST-REGIONS-INNER-SOLVER               '/
      data description( 6) /'NEWTON-LINE-SEARCH-INNER-SOLVER          '/
      data description( 7) /'TRUNCATED-NEWTON-LINE-SEARCH-INNER-SOLVER'/
      data description( 8) /'LINEAR-SYSTEMS-SCALING-AVOIDED           '/
      data description( 9) /'FIXED-VARIABLES-REMOVAL-AVOIDED          '/
      data description(10) /'ADD-SLACKS                               '/
      data description(11) /'OBJECTIVE-AND-CONSTRAINTS-SCALING-AVOIDED'/
      data description(12) /'IGNORE-OBJECTIVE-FUNCTION                '/
      data description(13) /'FEASIBILITY-TOLERANCE                    '/
      data description(14) /'OPTIMALITY-TOLERANCE                     '/
      data description(15) /'ACC-FEASIBILITY-THRESHOLD                '/
      data description(16) /'ACC-OPTIMALITY-THRESHOLD                 '/
      data description(17) /'OUTPUT-DETAIL                            '/
      data description(18) /'NCOMP-ARRAY                              '/

      data addinfo( 1) /' '/
      data addinfo( 2) /' '/
      data addinfo( 3) /' '/
      data addinfo( 4) /' '/
      data addinfo( 5) /' '/
      data addinfo( 6) /' '/
      data addinfo( 7) /' '/
      data addinfo( 8) /' '/
      data addinfo( 9) /' '/
      data addinfo(10) /' '/
      data addinfo(11) /' '/
      data addinfo(12) /' '/
      data addinfo(13) /'D'/
      data addinfo(14) /'D'/
      data addinfo(15) /'D'/
      data addinfo(16) /'D'/
      data addinfo(17) /'I'/
      data addinfo(18) /'I'/

C     EXTERNAL FUNCTIONS
      external lss,scl

C     INITIALIZATION
      if ( lss(lsssub) ) then
          lssdes = lsssub
          if ( lsssub .eq. 'MA57' ) then
              scldes = lsssub
          else
              if ( scl(sclsub) ) then
                  scldes = sclsub
              else
                  scldes = 'NONE'
              end if
          end if
      else
          lssdes = 'NONE'
          scldes = 'NONE'
      end if

C     BANNER
      if ( iprintctl(1) ) then
C          write(*, 8000)
          write(10,8000)
      end if

C     OPENING THE SPECIFICATION FILE
      open(20,err=300,file='algencan.dat',status='old')

      if ( iprintctl(2) ) then
C          write(*, 9005)
          write(10,9005)
      end if

C     MAIN LOOP

 100  continue

C     READING LINES
      read(20,fmt=1000,err=400,end=200) line

C     PROCESS LINES

C     Find first character
      i = 1
 110  if ( i .le. 80 .and. line(i:i) .eq. ' ' ) then
          i = i + 1
          go to 110
      end if

C     Skip blank lines
      if ( i .gt. 80 ) then
          go to 100
      end if

      ifirst = i

C     Skip comments
      if ( line(ifirst:ifirst) .eq. '*' .or.
     +     line(ifirst:ifirst) .eq. '#' ) then
          go to 100
      end if

C     Find the end of the keyword
      i = ifirst + 1
 120  if ( i .le. 80 .and. line(i:i) .ne. ' ' ) then
          i = i + 1
          go to 120
      end if

      ilast = i - 1

C     Obtain the first 10 characters and convert to upper-case
      keyword = '          '
      do i = 1,min( 10, ilast - ifirst + 1 )
          keyword(i:i) = line(ifirst+i-1:ifirst+i-1)
          do j = 1,26
              if ( keyword(i:i) .eq. lower(j) ) then
                  keyword(i:i) = upper(j)
              end if
          end do
      end do

C     Look up the keyword in the dictionary
      i = 1
 130  if ( i .le. nwords .and. keyword .ne. dictionary(i) ) then
          i = i + 1
          go to 130
      end if

C     Ignore unknown keywords
      if ( i .gt. nwords ) then
          if ( iprintctl(2) ) then
C              write(*, 9020) line(ifirst:ilast)
              write(10,9020) line(ifirst:ilast)
          end if
          go to 100
      end if

      ikey = i

C     Read additional information if needed
      if ( addinfo(ikey) .ne. ' ' ) then

C         Skip blanks
          i = ilast + 1
 140      if ( i .le. 80 .and. line(i:i) .eq. ' ' ) then
              i = i + 1
              go to 140
          end if

C         Ignore keywords without the required information
          if ( i .gt. 80 ) then
              if ( iprintctl(2) ) then
C                  write(*, 9030) description(ikey)
                  write(10,9030) description(ikey)
              end if
              go to 100
          end if

C         Read additional information
          if ( addinfo(ikey) .eq. 'I' ) then
              read(unit=line(i:80),fmt=2000) inum

          else if ( addinfo(ikey) .eq. 'D' ) then
              read(unit=line(i:80),fmt=3000) dnum
          end if

      end if

C     Process keyword
      if ( iprintctl(2) ) then
          if ( addinfo(ikey) .eq. ' ' ) then
C              write(*, 9040) description(ikey)
              write(10,9040) description(ikey)
          else if ( addinfo(ikey) .eq. 'I' ) then
C              write(*, 9041) description(ikey),inum
              write(10,9041) description(ikey),inum
          else if ( addinfo(ikey) .eq. 'D' ) then
C              write(*, 9042) description(ikey),dnum
              write(10,9042) description(ikey),dnum
          end if
      end if

C     Set the corresponding algencan argument
      if ( ikey .eq.  1 ) then
          hptype = 'INCQUO'

      else if ( ikey .eq.  2 ) then
          hptype = 'HAPPRO'

      else if ( ikey .eq.  3 ) then
          if ( seconde .or. truehpr ) then
              hptype = 'TRUEHP'
          else
              if ( iprintctl(2) ) then
C                  write(* ,9100) description(ikey)
                  write(10,9100) description(ikey)
              end if
          end if

      else if ( ikey .eq.  4 ) then
          skipacc = .true.

      else if ( ikey .eq.  5 ) then
          if ( .not. seconde ) then
              if ( iprintctl(2) ) then
C                  write(* ,9110) description(ikey)
                  write(10,9110) description(ikey)
              end if
          else if ( .not. lss(lsssub) ) then
              if ( iprintctl(2) ) then
C                  write(* ,9120) description(ikey)
                  write(10,9120) description(ikey)
              end if
          else
              innslvr = 'TR'
          end if

      else if ( ikey .eq.  6 ) then
          if ( .not. seconde ) then
              if ( iprintctl(2) ) then
C                  write(* ,9110) description(ikey)
                  write(10,9110) description(ikey)
              end if
          else if ( .not. lss(lsssub) ) then
              if ( iprintctl(2) ) then
C                  write(* ,9120) description(ikey)
                  write(10,9120) description(ikey)
              end if
          else
              innslvr = 'NW'
          end if

      else if ( ikey .eq.  7 ) then
          innslvr = 'TN'

      else if ( ikey .eq.  8 ) then
          if ( lss(lsssub) ) then
              sclsys = .false.

          else
              if ( iprintctl(2) ) then
C                  write(* ,9130) description(ikey)
                  write(10,9130) description(ikey)
              end if
          end if

      else if ( ikey .eq.  9 ) then
          rmfixv = .false.

      else if ( ikey .eq. 10 ) then
          slacks = .true.

      else if ( ikey .eq. 11 ) then
          scale = .false.

      else if ( ikey .eq. 12 ) then
          ignoref = .true.
          scale   = .false.

      else if ( ikey .eq. 13 ) then
          epsfeas = dnum

      else if ( ikey .eq. 14 ) then
          epsopt = dnum

      else if ( ikey .eq. 15 ) then
          efacc  = dnum

      else if ( ikey .eq. 16 ) then
          eoacc  = dnum

      else if ( ikey .eq. 17 ) then
          iprint = inum

      else if ( ikey .eq. 18 ) then
          ncomp = inum
      end if

C     IIERATE
      go to 100

C     END OF LOOP

C     TERMINATIONS

C     CLOSING SPECIFICATION FILE
 200  continue
      close(20)
      go to 500

C     NO SPECIFICATION FILE
 300  continue
      if ( iprintctl(2) ) then
C          write(*, 9000)
          write(10,9000)
      end if
      go to 500

C     ERROR READING THE SPECIFICATION FILE
 400  continue
      if ( iprintctl(2) ) then
C          write(*, 9010)
          write(10,9010)
      end if
      go to 500

C     PRINTING PARAMETERS VALUES
 500  continue
      if ( iprintctl(2) ) then
C          write(* ,4000) firstde,seconde,truehpr,hptype,innslvr,sclsys,
C     +                   skipacc,lssdes,scldes,rmfixv,slacks,scale,
C     +                   epsfeas,epsopt,efacc,eoacc,iprint,ncomp
          write(10,4000) firstde,seconde,truehpr,hptype,innslvr,sclsys,
     +                   skipacc,lssdes,scldes,rmfixv,slacks,scale,
     +                   epsfeas,epsopt,efacc,eoacc,iprint,ncomp
      end if

C     NON-EXECUTABLE STATEMENTS

 1000 format(A80)
 2000 format(BN,I20)
 3000 format(BN,F24.0)
 4000 format(/,1X,'ALGENCAN PARAMETERS:',
     +       /,1X,'firstde  = ',     19X,L1,
     +       /,1X,'seconde  = ',     19X,L1,
     +       /,1X,'truehpr  = ',     19X,L1,
     +       /,1X,'hptype   = ',     14X,A6,
     +       /,1X,'innslvr  = ',     18X,A2,
     +       /,1X,'sclsys   = ',     19X,L1,
     +       /,1X,'skipacc  = ',     19X,L1,
     +       /,1X,'lsssub   = ',     16X,A4,
     +       /,1X,'sclsub   = ',     16X,A4,
     +       /,1X,'rmfixv   = ',     19X,L1,
     +       /,1X,'slacks   = ',     19X,L1,
     +       /,1X,'scale    = ',     19X,L1,
     +       /,1X,'epsfeas  = ',8X,1P,D12.4,
     +       /,1X,'epsopt   = ',8X,1P,D12.4,
     +       /,1X,'efacc    = ',8X,1P,D12.4,
     +       /,1X,'eoacc    = ',8X,1P,D12.4,
     +       /,1X,'iprint   = ',        I20,
     +       /,1X,'ncomp    = ',        I20)
 8000 format(/,1X,78('='),
     +       /,1X,'This is ALGENCAN 2.4.0.',
     +       /,1X,'ALGENCAN, an augmented Lagrangian method for ',
     +            'nonlinear programming, is part of',/,1X,'the TANGO ',
     +            'Project: Trustable Algorithms for Nonlinear ',
     +            'General Optimization.',/,1X,'See ',
     +            'http://www.ime.usp.br/~egbirgin/tango/ for details.',
     +       /,1X,78('='))
 9000 format(/,1X,'The optional specification file algencan.dat was ',
     +            'not found in the current',/,1X,'directory (this is ',
     +            'not a problem nor an error). The default values ',
     +            'for the',/,1X,'ALGENCAN parameters will be used.')
 9005 format(/,1X,'Specification file algencan.dat is being used.')
 9010 format(/,1X,'Error reading specification file algencan.dat.')
 9020 format(  1X,'Ignoring unknown keyword ',A41)
 9030 format(  1X,'Ignoring incomplete keyword ',A41)
 9040 format(1X,A41)
 9041 format(1X,A41,5X,I20)
 9042 format(1X,A41,1X,1P,D24.8)

 9100 format(/,1X,'Warning: Ignoring keyword ',A41,'.',
     +       /,1X,'This option requires subroutines EVALJAC, EVALH ',
     +            'and EVALHC, or, alternatively, subroutines ',
     +            'EVALGJAC and EVALHL, or, alternatively, EVALGJACP ',
     +            'and EVALHLP, to be coded by the user. If you ',
     +            'already coded them, set array CODED in subrutine ',
     +            'INIP appropiately.',/)
 9110 format(/,1X,'Warning: Ignoring keyword ',A41,'.',
     +       /,1X,'This option requires subroutines EVALJAC, EVALH ',
     +            'and EVALHC, or, alternatively,',/,1X,'subroutines ',
     +            'EVALGJAC and EVALHL, to be coded by the user. If ',
     +            'you already coded them, set array CODED in ',
     +            'subrutine INIP appropiately.',/)
 9120 format(/,1X,'Warning: Ignoring keyword ',A41,'.',
     +       /,1X,'This option requires subroutine MA27 or MA57 from ',
     +            'HSL to be provided by the',/,1X,'user. If you ',
     +            'have any of them, see the compilation instructions ',
     +            'for details.',/)
 9130 format(/,1X,'Warning: Ignoring keyword ',A41,'.',
     +       /,1X,'This option applies only to the case in which ',
     +            'subroutines MA27 or MA57 from HSL are being used ',
     +            'to solve linear systems. If linear systems are ',
     +            'being solved and scaled, either the embedded ',
     +            'scaling option of MA57 or one of the subroutines ',
     +            'MC30 or MC77 from HSL would be used in conjuction ',
     +            'with MA27.')

      end
