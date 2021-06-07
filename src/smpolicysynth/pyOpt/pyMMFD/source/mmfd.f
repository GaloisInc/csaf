      SUBROUTINE MMFD (IIOPT,IIONED,IIPRINT,NDV,NCON,XX,XL,XU,
     1 OBJ,G,IDG,WK,NRWK,IWK,NRIWK,IFILE,CT,CTMIN,DABOBJ,
     2 DELOBJ,THETAZ,PMLT,ITMAX,ITRMOP,NFUN,NGRD,OBJFUN,OBJGRD)
C
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
C
      DIMENSION XX(NDV),XL(NDV),XU(NDV),G(NCON),IDG(NCON),
     1 B(NDV,NCON),IC(NCON),DF(NDV+1),WK(NRWK),IWK(NRIWK),DDF(NDV)
      DIMENSION X(NDV+1),VLB(NDV+1),VUB(NDV+1),A(NDV+1,NCON+NDV)
      CHARACTER*(*) IFILE
      EXTERNAL OBJFUN,OBJGRD
C     
C  PARAMETERS
C
      ISTRAT=0
      IF (IIOPT.EQ.0) IOPT=5
      IF (IIOPT.EQ.1) IOPT=4
      IF (IIONED.EQ.0) IONED=7
      IF (IIONED.EQ.1) IONED=8
      IF (IIONED.EQ.2) IONED=6
      IF (IIONED.EQ.3) IONED=5
      IF (IIPRINT.EQ.0) IPRINT=0000
      IF (IIPRINT.EQ.1) IPRINT=0010
      IF (IIPRINT.EQ.2) IPRINT=0030
      IGRAD=1
      DO 10 I=1,NDV
      X(I)=XX(I)
      VLB(I)=XL(I)
10    VUB(I)=XU(I)
      NRA=NDV
      NCOLA=NCON
C 
C  OPEN WRITE FILE
C     
      IF (IIPRINT.EQ.0) GO TO 20
      OPEN(UNIT=6,FILE=IFILE,STATUS='UNKNOWN')
20    CONTINUE
C
C  INITIALIZE 
C
      INFO=-2
      CALL ADS(INFO,ISTRAT,IOPT,IONED,IPRINT,IGRAD,NDV,NCON,X,
     1VLB,VUB,OBJ,G,IDG,NGT,IC,DF,A,NRA,NCOLA,WK,NRWK,IWK,NRIWK)
      WK(3)=CT
      WK(6)=CTMIN
      WK(8)=DABOBJ
      WK(12)=DELOBJ
      WK(35)=THETAZ
      WK(38)=PMLT
      IWK(3)=ITMAX
      IWK(4)=ITRMOP
C      
      NFUN=0
      NGRD=0
C
C  OPTIMIZE 
C
30	  CALL ADS(INFO,ISTRAT,IOPT,IONED,IPRINT,IGRAD,NDV,NCON,X,
     1VLB,VUB,OBJ,G,IDG,NGT,IC,DF,A,NRA,NCOLA,WK,NRWK,IWK,NRIWK)
C
C  EVALUATIONS
C
	  IF (INFO.EQ.0) GO TO 80
      DO 35 I=1,NDV
35    XX(I)=X(I)
	  IF (INFO.EQ.1) GO TO 40
	  IF (INFO.EQ.2) GO TO 50
C
C  EVALUATE OBJECTIVE AND CONSTRAINTS
C
40    CALL OBJFUN(NDV,NCON,XX,OBJ,G)
      NFUN=NFUN+1
	  GO TO 30
C
C  EVALUATE GRADIENTS
C 
50    CONTINUE
C  GRADIENT OF OBJECTIVE AND CONSTRAINTS
      CALL OBJGRD(NDV,NCON,XX,OBJ,G,DDF,B)
      DO 55 I=1,NDV
55    DF(I)=DDF(I)
      NGRD=NGRD+1
C  STORE APPROPRIATE GRADIENTS IN ARRAY A.
	  IF (NGT.EQ.0) GO TO 30
	  DO 60 J=1,NGT
	  K=IC(J)
      DO 60 I=1,NDV
60    A(I,J)=B(I,K)
	    GO TO 30
C
80    CONTINUE
C
C  PRINT FINAL RESULTS 
C
      IF (IIPRINT.EQ.0) GO TO 90
      WRITE(6,1650) NFUN
      WRITE(6,1750) NGRD
90    CONTINUE
C
C  OUTPUT HANDLING
C
      DO 100 I=1,NDV
100   XX(I)=X(I)
C
      RETURN
C  ------------------------------------------------------------------
C  FORMATS
C  ------------------------------------------------------------------
1650  FORMAT(//8X,'NUMBER OF FUNC-CALLS:  NFUN =',I5)
1750  FORMAT(/8X,'NUMBER OF GRAD-CALLS:  NGRD =',I5)
C
      END
