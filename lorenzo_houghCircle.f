      program ht_circle
      

      parameter(NHMAX=100)
      parameter(Amin=-50000.,Amax=50000.)
      parameter(Bmin=-50000.,Bmax=50000.)
      parameter(Rmin=0.,Rmax=500000.)
      parameter(Nbin=100,Nhit=100)
      real xh(Nhit),yh(Nhit)
      integer accu(Nbin,Nbin,Nbin)

c      Rmin=0
c      Rmax=sqrt(Amax**2+Bmax**2)


c    azzeramento matrice di accumulazione
      
      do i=1,Nbin
         do j=1,Nbin
            do k=1,Nbin
               accu(i,j,k)=0
            enddo   
         enddo   
      enddo
      

c     dimensione dei bins in A e B

      dA=(Amax-Amin)/Nbin
      dB=(Bmax-Bmin)/Nbin
      dR=(Rmax-Rmin)/Nbin

c Riempio la matrice di accumulazione
            

      open(unit=1,file="../datafiles/hits-1.txt")      
      iread=0
      nhits=0
      do ihit=1,NHMAX
         read(1,*,end=10) x,y
         print*, x/y
         do iA=1,Nbin
            a=Amin+(ia+0.5)*dA
            do iB=1,Nbin
               b=Bmin+(ib+0.5)*dB
               R=sqrt((x-a)**2+(y-b)**2)
               iR=int(nbin*(R-Rmin)/(Rmax-Rmin))+1
               if(R.lt.Rmax)then
                  accu(iA,iB,iR)=accu(iA,iB,iR)+1  
               endif   
            enddo
         enddo   
      enddo
      


 10   close(1)




      accumax=-1

      do iA=1,Nbin
        do iB=1,Nbin
            do iR=1, Nbin
               if (accu(iA,iB,iR) .ge. accumax) then
                  accumax=accu(iA,iB,iR)
                  iAMAX=iA
                  iBMAX=iB
                  iRMAX=IR
                  aa=amin+iA*dA
                  bb=bmin+iB*dB
                  rr=rmin+IR*dR
c                  print*,accu(iA,iB,iR),aa,bb,rr
               endif
            enddo
         enddo   
      enddo
      

      print*,accumax,iAMAX,iBMAX,aa,bb,rr

 100  format(f10.5,f10.5)

 

      end
