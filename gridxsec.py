#!/usr/bin/env python
import numpy

#TODO: Add Sections going along other things than great circles..
#TODO: Gather section methods and make subclasses

class SectionError(Exception):
   pass


class SectionBase(object) :

   @property 
   def mask(self) :
      return self._mask

   @property 
   def mask2(self) :
      return self._mask2

   @property 
   def flagu(self) :
      return self._flagu

   @property 
   def flagv(self) :
      return self._flagv

   @property 
   def grid_indexes(self) :
      return self._section_i,self._section_j

   @property 
   def distance(self) :
      return self._distance_along_section

   @property 
   def longitude(self) :
      return self._section_longitudes

   @property 
   def latitude(self) :
      return self._section_latitudes

   def plot_section_arrows(self,plon=None,plat=None) :
      import matplotlib
      fig = matplotlib.pyplot.figure(figsize=(12,12))
      ax=fig.add_subplot(111)
      ax.hold(True)
      ax.pcolor(self._mask)
      i=numpy.arange(self._flagu.shape[1]) - 0.5
      j=numpy.arange(self._flagu.shape[0])
      x,y=numpy.meshgrid(i,j)
      I=numpy.where(self._flagu<>0)
      ax.quiver(x[I],y[I],self._flagu[I],numpy.zeros(self._flagu.shape)[I],width=.002)
      i=numpy.arange(self._flagu.shape[1])
      j=numpy.arange(self._flagu.shape[0]) - 0.5
      x,y=numpy.meshgrid(i,j)
      I=numpy.where(self._flagv<>0)
      ax.quiver(x[I],y[I],numpy.zeros(self._flagv.shape)[I], self._flagv[I],width=.002)
      ax.set_title("positive direction across section")
      self._add_gridlines(ax,plon,plat) 
      return fig


   def plot_section_mask(self,plon=None,plat=None) :
      import matplotlib
      fig = matplotlib.pyplot.figure(figsize=(12,24))
      ax=fig.add_subplot(211)
      J,I=numpy.where(self._flagu<>0)
      ax.scatter(I,J,30,self._flagu[J,I],edgecolors="face")
      self._add_gridlines(ax,plon,plat) 
      ax.set_title("u-flag; negative values mean negative\n grid direction is treated as positive direction")
      ax=fig.add_subplot(212)
      J,I=numpy.where(self._flagv<>0)
      ax.scatter(I,J,30,self._flagv[J,I],edgecolors="face")
      self._add_gridlines(ax,plon,plat) 
      ax.set_title("v-flag; negative values mean negative\n grid direction is treated as positive direction")
      return fig


   def plot_section_1d(self,plon=None,plat=None) :
      import matplotlib
      fig = matplotlib.pyplot.figure(figsize=(12,18))
#
      ax=fig.add_subplot(321)
      ax.plot(self._distance_along_section,self._section_longitudes)
      ax.set_title("longitude along section")
#
      ax=fig.add_subplot(322)
      ax.plot(self._distance_along_section,self._section_latitudes)
      ax.set_title("latitude along section")
#
      ax=fig.add_subplot(323)
      ax.plot(self._distance_along_section,self._section_i)
      ax.set_title("i pivot along section")
#
      ax=fig.add_subplot(324)
      ax.plot(self._distance_along_section,self._section_j)
      ax.set_title("j pivot along section")
#
      ax=fig.add_subplot(325)
      ax.plot(self._distance_along_section,self._distance_along_section_1)
      ax.set_title("distance measure 1")
#
      ax=fig.add_subplot(326)
      ax.plot(self._distance_along_section,self._distance_along_section_2)
      ax.set_title("distance measure 2")
#
      return fig


   def _add_gridlines(self,ax,plon,plat) :
      if plon is not None :
         CS=ax.contour(plon,numpy.arange(-180,180,10),colors="k")
         ax.clabel(CS, inline=1, fontsize=10,fmt="%1.1f")
      if plat is not None :
         CS = ax.contour(plat,numpy.arange(-90,90,10),colors="k")
         ax.clabel(CS, inline=1, fontsize=10,fmt="%1.1f")


class Section(SectionBase) :
   def __init__(self,waypoints_lon,waypoints_lat,grid_lon,grid_lat) :

      self._jdm,self._idm = grid_lon.shape

      if len(waypoints_lon) <> 2 or len(waypoints_lat) <> 2 :
         raise SectionError,"Only two waypoints all2oed"

      ########### The following assumes xsections go along great circles ###################

      # Radius vectors for start and end points
      self._waypoints_lon=waypoints_lon
      self._waypoints_lat=waypoints_lat
      self._waypoint_vectors=geo2cart(self._waypoints_lon,self._waypoints_lat)

      # Normal vector of plane defined by cross product of
      # 1) Vector from earth center to start of section
      # 2) Vector from earth center to end   of section
      self._normal_vector=numpy.cross(self._waypoint_vectors[0,:],self._waypoint_vectors[1,:])

      # The section plane is indeterminate if waypoints and origo is on the same line. Abort
      if numpy.sqrt(numpy.sum(self._normal_vector**2)) < 1e-2:
         raise SectionError,"Section is on opposite sides of the earth"

      # Make normal vector a unit vector
      self._normal_vector= self._normal_vector / numpy.sqrt(numpy.sum(self._normal_vector**2))

      # Radius vector to all grid points
      rvec=geo2cart(grid_lon,grid_lat)

      # Now go through grid and mark all points on one side of the great circle
      tmp=dot_product_last( self._normal_vector,rvec)
      self._mask = numpy.where(tmp < 0,1,0)

      self._flagu=numpy.zeros(grid_lon.shape)
      self._flagv=numpy.zeros(grid_lon.shape)
      J,I=numpy.where(self._mask==1)

      # TODO: handleperiodic grids
      Ip1 = numpy.minimum(I+1,self._idm-1)
      Jp1 = numpy.minimum(J+1,self._jdm-1)

      # Now calculate the node points along the hemisphere line by  using a ``telescopic'' sum
      self._flagu[J,I  ] = self._flagu[J,I  ] + 1
      self._flagu[J,Ip1] = self._flagu[J,Ip1] - 1
      self._flagv[J,I  ] = self._flagv[J,I  ] + 1
      self._flagv[Jp1,I] = self._flagv[Jp1,I] - 1

      # Reduce the number of points to those between section endpoints
      # Cross product first section point X grid vectors
      cp1=cross_product_last(self._waypoint_vectors[0],rvec)
      cp2=cross_product_last(rvec,self._waypoint_vectors[1])
      #self._mask2 = dot_product_last(cp1,cp2)
      tmp  = dot_product_last(cp1,self._normal_vector)
      tmp2 = dot_product_last(cp2,self._normal_vector)
      self._mask2 = numpy.minimum(tmp,tmp2)
    
      # Modify mask and u/v flags
      self._flagu[self._mask2<0.] = 0.
      self._flagv[self._mask2<0.] = 0.
      self._mask [self._mask2<0.] = 0.

      # Remove boundary points
      # TODO: handleperiodic grids
      self._flagu[0,:]=0
      self._flagu[-1,:]=0
      self._flagu[:,0]=0
      self._flagu[:,-1]=0
      self._flagv[0,:]=0
      self._flagv[-1,:]=0
      self._flagv[:,0]=0
      self._flagv[:,-1]=0

      # Find pivot points along section. This is the simplest approach where we use cell centers 
      # TODO: find actual crossing points of grid ?
      J,I = numpy.where(numpy.logical_or(self._flagu<>0,self._flagv<>0))
      self._section_i=I
      self._section_j=J
      self._section_longitudes = grid_lon[J,I]
      self._section_latitudes  = grid_lat[J,I]
      self._section_vectors    = rvec    [J,I,:]

      # Approach 1) Haversine for distance. TODO: Will not work when section > 180 degrees...
      self._distance_along_section_1 = [ haversine(e[0],e[1],self._waypoints_lon[0],self._waypoints_lat[0]) for e in 
            zip(self._section_longitudes,self._section_latitudes) ]
      self._distance_along_section_1 = numpy.array(self._distance_along_section_1)

      # Approach 2) Angle in projected plane
      #x-coord in projected plane:
      xvec = self._waypoint_vectors[0]

      #y-coord in projected plane:
      yvec = numpy.cross(self._normal_vector,xvec) # TODO: check if on same line with origo!

      # Projection of section vectors onto plane
      xcomp = dot_product_last(self._section_vectors,xvec)
      ycomp = dot_product_last(self._section_vectors,yvec)

      # Angle in plane
      angle = numpy.arctan2(ycomp,xcomp)

      # Let angle be in 0 to 2pi
      angle[angle<0] = angle[angle<0] + 2.*numpy.pi
      self._distance_along_section_2 = angle * 6371000
      
      self._distance_along_section = numpy.copy(self._distance_along_section_2)
      I=numpy.argsort(self._distance_along_section)
      self._section_i         =self._section_i[I]
      self._section_j         =self._section_j[I]
      self._section_longitudes=self._section_longitudes[I]
      self._section_latitudes =self._section_latitudes[I]
      self._distance_along_section_1 =self._distance_along_section_1[I]
      self._distance_along_section_2 =self._distance_along_section_2[I]
      self._distance_along_section   =self._distance_along_section[I]



#      return self._flagv

class SectionIJSpace(SectionBase) :
   def __init__(self,waypoints_i,waypoints_j,grid_lon,grid_lat) :

      jishape = grid_lon.shape
      self._jdm,self._idm = jishape

      if len(waypoints_i) <> 2 or len(waypoints_j) <> 2 :
         raise SectionError,"Only two waypoints all2oed"

      self._waypoints_x=waypoints_i
      self._waypoints_y=waypoints_j

      # Waypoint vector in IJ-plane
      v1=numpy.array([self._waypoints_x[1]-self._waypoints_x[0],self._waypoints_y[1]-self._waypoints_y[0],0])

      # Vector from first point to all I,J points
      Jg,Ig=numpy.meshgrid(range(jishape[1]),range(jishape[0]))
      Jg=Jg.flatten()
      Ig=Ig.flatten()
      v2=numpy.zeros((Ig.size,3))
      v3=numpy.zeros((Ig.size,3))
      v2[:,0]=Ig-self._waypoints_x[0]
      v2[:,1]=Jg-self._waypoints_y[0]
      v2[:,2]=0.
      v3[:,0]=Ig-self._waypoints_x[1]
      v3[:,1]=Jg-self._waypoints_y[1]
      v3[:,2]=0.

      # Angle all points in IJ space and vector in 
      self._normal_vector=numpy.cross(v1,v2)

      # Now go through grid and mark all points on one side of waypoint vector
      #self._mask = numpy.where(self._normal_vector[:,2] < 0,1,0)
      self._mask = numpy.where(self._normal_vector[:,2] < 0,True,False)
      self._mask.shape=tuple(jishape)


      # TODO: handleperiodic grids
      self._flagu=numpy.zeros(self._mask.shape)
      self._flagv=numpy.zeros(self._mask.shape)
      J,I=numpy.where(self._mask)
      Ip1 = numpy.minimum(I+1,self._idm-1)
      Jp1 = numpy.minimum(J+1,self._jdm-1)
      self._flagu[J,I  ] = self._flagu[J,I  ] + 1
      self._flagu[J,Ip1] = self._flagu[J,Ip1] - 1
      self._flagv[J,I  ] = self._flagv[J,I  ] + 1
      self._flagv[Jp1,I] = self._flagv[Jp1,I] - 1

      # Remove boundary points
      # TODO: handleperiodic grids
      self._flagu[0,:]=0
      self._flagu[-1,:]=0
      self._flagu[:,0]=0
      self._flagu[:,-1]=0
      self._flagv[0,:]=0
      self._flagv[-1,:]=0
      self._flagv[:,0]=0
      self._flagv[:,-1]=0

      # Reduce the number of points to those between section endpoints
      tmp  = dot_product_last(v1,v2)
      tmp2 = dot_product_last(v3,-v1)
      self._mask2 = tmp*tmp2 > 0.
      self._mask2.shape=tuple(jishape)

      # Modify mask and u/v flags
      self._flagu[~self._mask2] = 0.
      self._flagv[~self._mask2] = 0.
      #self._mask [~self._mask2] = 0.

      # Find pivot points along section. This is the simplest approach where we use cell centers 
      # TODO: find actual crossing points of grid ?
      I,J = numpy.where(numpy.logical_or(self._flagu<>0,self._flagv<>0))
      self._section_i=I
      self._section_j=J

      # Approach 1) Haversine for distance. TODO: Will not work when section > 180 degrees...
      self._section_longitudes     = grid_lon[J,I]
      self._section_latitudes      = grid_lat[J,I]
      #
      self._distance_along_section = numpy.sqrt((self._section_i-self._waypoints_x[0])**2 + (self._section_j-self._waypoints_y[0])**2) # Index space
      I=numpy.argsort(self._distance_along_section)
      self._section_i              = self._section_i[I]
      self._section_j              = self._section_j[I]
      self._distance_along_section = self._distance_along_section[I]
      self._section_longitudes     = self._section_longitudes[I]
      self._section_latitudes      = self._section_latitudes[I]


      # Cumulative distance (nb : includes zig-zag)
      self._distance_along_section = numpy.zeros(I.size)
      for k in range(1,I.size) :
         self._distance_along_section[k] = self._distance_along_section[k-1] + haversine(
               self._section_longitudes[k]  ,self._section_latitudes[k]  ,
               self._section_longitudes[k-1],self._section_latitudes[k-1]
               )



#
#
#
#P=m.pcolor(x,y,sec.mask)
#m.drawcoastlines()
#pl=m.drawparallels(numpy.arange(-90.,120.,10.),labels=[1,0,0,0]) # draw parallels
#mer=m.drawmeridians(numpy.arange(0.,420.,10.),labels=[0,0,0,1]) # draw meridians



def geo2cart(lon,lat) :
   deg2rad = numpy.pi / 180.
   lmbda=numpy.array(lat)*deg2rad
   theta=numpy.array(lon)*deg2rad

   tmp1=numpy.cos(lmbda)*numpy.cos(theta)
   tmp2=numpy.cos(lmbda)*numpy.sin(theta)
   tmp3=numpy.sin(lmbda)
   tmp1.shape = tuple(list(tmp1.shape)+[1])
   tmp2.shape = tuple(list(tmp2.shape)+[1])
   tmp3.shape = tuple(list(tmp3.shape)+[1])
   t=numpy.concatenate((tmp1,tmp2,tmp3),axis=-1)
   return t



def cross_product_last(v1,v2) :
   tmp1 =  v1[...,1] * v2[...,2] - v1[...,2]*v2[...,1]
   tmp2 =  v1[...,2] * v2[...,0] - v1[...,0]*v2[...,2]
   tmp3 =  v1[...,0] * v2[...,1] - v1[...,1]*v2[...,0]
   tmp1.shape = tuple(list(tmp1.shape)+[1])
   tmp2.shape = tuple(list(tmp2.shape)+[1])
   tmp3.shape = tuple(list(tmp3.shape)+[1])
   t=numpy.concatenate((tmp1,tmp2,tmp3),axis=-1)
   #print "cross_product_last :",v1.shape,v2.shape,t.shape
   return t



def dot_product_last(v1,v2) :
   #print "dot_product_last :",v1.shape,v2.shape
   tmp  =  v1[...,0] * v2[...,0]
   tmp +=  v1[...,1] * v2[...,1]
   tmp +=  v1[...,2] * v2[...,2]
   return tmp

def haversine(lon1,lat1,lon2,lat2) :
   deg2rad = numpy.pi / 180.
   dlon=lon2-lon1
   dlat=lat2-lat1
   dlon=dlon*deg2rad
   dlat=dlat*deg2rad
   #a=numpy.sin(dlat/2.)**2 + numpy.cos(lon1*deg2rad) * numpy.cos(lon2*deg2rad) * numpy.sin(dlon/2.)**2
   a=numpy.sin(dlat/2.)**2 + numpy.cos(lat1*deg2rad) * numpy.cos(lat2*deg2rad) * numpy.sin(dlon/2.)**2
   #c=2.*numpy.arctan2(numpy.sqrt(a),numpy.sqrt(1.-a))
   c=2.*numpy.arcsin(numpy.sqrt(a))
   d=6371000.*c
   return d




