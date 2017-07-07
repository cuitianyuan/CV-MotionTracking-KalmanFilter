"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2


from numpy.random import uniform,randn
from numpy.linalg import norm
import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods. Refer to the method
    run_particle_filter( ) in experiment.py to understand how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles. This should be a N x 2 array where
                                        N = self.num_particles. This component is used by the autograder so make sure
                                        you define it appropriately.
        - self.weights (numpy.array): Array of N weights, one for each particle.
                                      Hint: initialize them with a uniform normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video frame that will be used as the template to
                                       track.
        - self.frame (numpy.array): Current video frame from cv2.VideoCapture().

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame, values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track, values in [0, 255].
            kwargs: keyword arguments needed by particle filter model, including:
                    - num_particles (int): number of particles.
                    - sigma_mse (float): sigma value used in the similarity measure.
                    - sigma_dyn (float): sigma value that can be used when adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y, width, and height values.
        """

        self.num_particles = kwargs.get('num_particles')  # required by the autograder
#        self.sigma_exp = kwargs.get('sigma_mse')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp') # required by the autograder

        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        # Todo: Initialize your particles array. Read the docstring.
        self.particles = np.column_stack( (uniform(template.shape[0]/2, frame.shape[0]-template.shape[0]/2, size=self.num_particles)#.reshape((self.num_particles, 1)) 
                                           ,uniform(template.shape[1]/2, frame.shape[1]-template.shape[1]/2, size=self.num_particles)#.reshape((self.num_particles, 1)) 
                                          )
                                          ) 
#        particles = np.column_stack( (uniform(template.shape[0]/2, frame.shape[0]-template.shape[0]/2, size=num_particles) #.reshape((num_particles, 1)) 
#                                           ,uniform(template.shape[1]/2, frame.shape[1]-template.shape[1]/2, size=num_particles)#.reshape((num_particles, 1)) 
#                                          )
#                                         ) 
#        particles = np.column_stack( (np.random.randint(template.shape[0]/2, frame.shape[0]-template.shape[0]/2, size=self.num_particles) 
#                                    , np.random.randint(template.shape[1]/2, frame.shape[1]-template.shape[1]/2, size=self.num_particles) 
#                                    ))  
#        
#        particles = np.column_stack( (np.random.randint(template.shape[0]/2, frame.shape[0]-template.shape[0]/2, size=num_particles) 
#                                    , np.random.randint(template.shape[1]/2, frame.shape[1]-template.shape[1]/2, size=num_particles) 
#                                    ))  
#                                          
#                                          
#        weights =   np.ones( num_particles) / num_particles

        # Todo: Initialize your weights array. Read the docstring.
        self.weights =   np.ones(self.num_particles) /self.num_particles
        # Initialize any other components you may need when designing your filter.



    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.

        """

        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """

        return self.weights
    
    
#         particles = np.column_stack( (uniform(template.shape[0]/2, frame.shape[0]-template.shape[0]/2, size=num_particles) #.reshape((num_particles, 1)) 
#                                                   ,uniform(template.shape[1]/2, frame.shape[1]-template.shape[1]/2, size=num_particles)#.reshape((num_particles, 1)) 
#                                                  )
#                                                 ) 
#         weights =   np.ones( num_particles) / num_particles

   
    def predict(self,particles, sigma_dyn, frame, template):
        newP =  particles + randn(len(particles),2)*sigma_dyn 
         
        newP[newP[:,0] > frame.shape[0]-template.shape[0]/2-1, 0] = frame.shape[0]-template.shape[0]/2-1
#        print newP[newP[:,0] > frame.shape[0]-template.shape[0]/2 , 0] 
        newP[newP[:,0] < template.shape[0]/2+1, 0]  = template.shape[0]/2
#        print newP[newP[:,0] > frame.shape[0]-template.shape[0]/2 , 0] 
        newP[newP[:,1] > frame.shape[1]-template.shape[1]/2-1, 1]  = frame.shape[1]-template.shape[1]/2-1
#        print newP[newP[:,0] > frame.shape[0]-template.shape[0]/2 , 0] 
        newP[newP[:,1] < template.shape[1]/2+1, 1]  = template.shape[1]/2+1
#        print newP[newP[:,0] > frame.shape[0]-template.shape[0]/2 , 0] 

#(newP[newP[:,0] > frame.shape[0]-template.shape[0]/2, 0]
#,newP[newP[:,0] < template.shape[0]/2, 0]
#, newP[newP[:,1] > frame.shape[1]-template.shape[1]/2, 1]
#, newP[newP[:,1] < template.shape[1]/2, 1])
#max(newP[:,0])
        self.particles =  newP
 
                      
#                      
#                      
    def update(self,particles, weights, frame, sigma_exp, template):
#        distance = []
#        frame_grey = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY ).astype(float) 
#        template_grey = cv2.cvtColor(template.astype(np.uint8), cv2.COLOR_BGR2GRAY ).astype(float)
#        
#        
#        for k in range(len(particles)):
#            lx = int( particles[k,0]-template.shape[0]/2)
#            ux = int( particles[k,0]-template.shape[0]/2)+ template.shape[0]  
#            ly = int( particles[k,1]-template.shape[1]/2)
#            uy = int( particles[k,1]-template.shape[1]/2)+ template.shape[1] 
#            pf = frame_grey[ lx:ux
#                            ,ly:uy    ]
#            distance.append( np.linalg.norm( pf  - template_grey )  )  #np.sqrt(np.sum((pf  - template_grey) **2))  
#
#        p = np.exp(-np.array(distance)/2.0/(sigma_exp**2))+1.e-300
#        print 'min dist', particles[ np.argmax(p/sum(p) ),:], distance[np.argmax(p/sum(p) )]
#        self.weights = p/sum(p) 




#        distance = []
#        frame_green =frame[:,:,1].astype(float) 
#        template_green = template[:,:,1].astype(float)
#        for k in range(len(particles)):
#            lx = int( particles[k,0]-template.shape[0]/2)
#            ux = int( particles[k,0]-template.shape[0]/2)+ template.shape[0]  
#            ly = int( particles[k,1]-template.shape[1]/2)
#            uy = int( particles[k,1]-template.shape[1]/2)+ template.shape[1] 
#            pf = frame_green[ lx:ux
#                            ,ly:uy    ]
#            mse = np.linalg.norm( pf  - template_green )
##            sc1 = int(template.shape[1]/10*np.random.rand())+1
##            sc2 =  template.shape[1]-sc1 
##            msec = np.linalg.norm( pf[:,sc1]  - pf[:,sc2] )
##            if msec<5.:
##                mse +=10000000
##                print k
#            distance.append(mse  )  #np.sqrt(np.sum((pf  - template_grey) **2))  
#
#        p = np.exp(-np.array(distance)/2.0/(sigma_exp**2))+1.e-300
#        print 'min dist', particles[ np.argmax(p/sum(p) ),:], distance[np.argmax(p/sum(p) )]
#        self.weights = p/sum(p) 





        distance = []
#        frame_green =frame[:,:,1].astype(float) 
#        template_green = template[:,:,1].astype(float)
        for k in range(len(particles)):
            lx = int( particles[k,0]-template.shape[0]/2)
            ux = int( particles[k,0]-template.shape[0]/2)+ template.shape[0]  
            ly = int( particles[k,1]-template.shape[1]/2)
            uy = int( particles[k,1]-template.shape[1]/2)+ template.shape[1] 
#            pf = frame_green[ lx:ux,ly:uy    ]
            mse = (np.linalg.norm(frame[lx:ux,ly:uy,1].astype(float) - template[:,:,1].astype(float) )
                        +np.linalg.norm(frame[lx:ux,ly:uy,0].astype(float) - template[:,:,0].astype(float) )
                        +np.linalg.norm(frame[lx:ux,ly:uy,2].astype(float) - template[:,:,2].astype(float) )
                        )
#            sc1 = int(template.shape[1]/10*np.random.rand())+1
#            sc2 =  template.shape[1]-sc1 
#            msec = np.linalg.norm( pf[:,sc1]  - pf[:,sc2] )
#            if msec<5.:
#                mse +=10000000
#                print k
            distance.append(mse  )  #np.sqrt(np.sum((pf  - template_grey) **2))  

        p = np.exp(-np.array(distance)/2.0/(sigma_exp*2))   #+1.e-300
#        print 'min dist', particles[ np.argmax(p/sum(p) ),:], distance[np.argmax(p/sum(p) )]
        self.weights = p/sum(p) 



#
##        
#        k = np.argmin(distance)  #np.argmax(p/sum(p))
#        print k
#        lx = int( particles[k,0]-template.shape[0]/2)
#        ux = int( particles[k,0]-template.shape[0]/2+ template.shape[0])  
#        ly = int( particles[k,1]-template.shape[1]/2)
#        uy = int( particles[k,1]-template.shape[1]/2+ template.shape[1])
#        pf = frame_green[ lx:ux,ly:uy    ]
#        plt.imshow(pf)
#        plt.imshow(frame)
#        plt.imshow(template_grey)
#        print 'dist', np.linalg.norm( pf  - template_grey )  # 4291.93126692
#        
#        
#        
#        particles[20,:] = [250,380] 
#         
#        
#        
#        k=20 
#        lx = int( particles[k,0]-template.shape[0]/2)
#        ux = int( particles[k,0]-template.shape[0]/2+ template.shape[0])  
#        ly = int( particles[k,1]-template.shape[1]/2)
#        uy = int( particles[k,1]-template.shape[1]/2+ template.shape[1])
#        pf = frame_grey[ lx:ux,ly:uy    ]
#        plt.imshow(pf)
#        print 'dist', np.linalg.norm( pf  - template_grey )  # 5261.91039452
#        
#        
#        
#        
#        im = np.zeros((720, 1280))
#        for k in range(len(particles)):
#            im[int(particles[k,0]),int(particles[k,1]) ] = (1./distance[k])/max( np.ones(1001)/distance )*255.
#        plt.imshow(im)
#        max(particles[:,0])
#        
#        # resample
#        s = np.random.choice(num_particles,num_particles,replace=True,p=weights)
#        particles = particles[s,]
#        weights = weights[s,]
#        
#        #predict
#        newP =  particles + randn(len(particles),2)*sigma_dyn 
#         
#        newP[newP[:,0] > frame.shape[0]-template.shape[0]/2, 0] = frame.shape[0]-template.shape[0]/2
#        newP[newP[:,0] < template.shape[0]/2, 0] = template.shape[0]/2
#        newP[newP[:,1] > frame.shape[1]-template.shape[1]/2, 1] = frame.shape[1]-template.shape[1]/2
#        newP[newP[:,1] < template.shape[1]/2, 1] = template.shape[1]/2
#        particles =  newP
        
        
    def resample( self):
        s = np.random.choice(self.num_particles,self.num_particles,replace=True,p=self.weights)
        self.particles = self.particles[s,]
        self.weights = self.weights[s,]/sum(self.weights[s,])
        
#        s = np.random.choice(num_particles,num_particles,replace=True,p=weights)
#        particles = particles[s,]
#        weights =  weights[s,]/sum( weights[s,])
         

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None (do not include a return call). This function
        should update the particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the image. This means you should address
        particles that are close to the image borders.

        Args:,
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        
        self.predict(self.particles, sigma_dyn=self.sigma_dyn , frame=self.frame, template=self.template )
        self.update(self.particles, self.weights, frame, sigma_exp=self.sigma_exp, template=self.template )
        self.resample(  )


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model updates here!
        These steps will calculate the weighted mean. The resulting values should represent the
        tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay each successive
        frame with the following elements:

        - Every particle's (u, v) location in the distribution should be plotted by drawing a
          colored dot point on the image. Remember that this should be the center of the window,
          not the corner.
        - Draw the rectangle of the tracking window associated with the Bayesian estimate for
          the current location which is simply the weighted mean of the (u, v) of the particles.
        - Finally we need to get some sense of the standard deviation or spread of the distribution.
          First, find the distance of every particle to the weighted mean. Next, take the weighted
          sum of these distances and plot a circle centered at the weighted mean with this radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the particle filter.
        """
       
#        frame_in = frame.copy()

#        particles = self.particles
#        weights = self.weights
#        template = self.template
#        

        u_weighted_mean = 0
        v_weighted_mean = 0
        for i in range(len(self.particles)):
            u_weighted_mean +=  self.particles[i, 0] *  self.weights[i]
            v_weighted_mean +=  self.particles[i, 1] *  self.weights[i]
        
        d = []
        for j in range( len(self.particles)):
            cv2.circle(frame_in, (int(self.particles[j, 1]), int(self.particles[j, 0]) ), 1, (0, 255, 0),  -1)
            d.append(np.linalg.norm(  self.particles[j, :]- np.array([u_weighted_mean,v_weighted_mean]) ) *  self.weights[j])  #find the distance of every particle to the weighted mean


#        'Draw the rectangle of the tracking window associated with the Bayesian estimate for
#          the current location which is simply the weighted mean of the (u, v) of the self.particles.  '''
        cv2.circle(frame_in, (int(v_weighted_mean), int(u_weighted_mean) ), 10, (0,0,255),  1)
        cv2.rectangle(frame_in
                       , (int(v_weighted_mean)-self.template.shape[1]/2, int(u_weighted_mean)-self.template.shape[0]/2) 
                       , (int(v_weighted_mean)+self.template.shape[1]/2, int(u_weighted_mean)+self.template.shape[0]/2)
                       , (0,0,255), 2)
        print 'circle radius', sum(d)
        cv2.circle(frame_in
                       ,  (int(v_weighted_mean), int(u_weighted_mean) )
                       , int(sum(d))
                       , (0,0,255), 1)
         




class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter object (parameters are the same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above. There is one element that is added
        called alpha which is explained in the problem set documentation. By calling super(...) all the elements used
        in ParticleFilter will be inherited so you do not have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        self.predict(self.particles, sigma_dyn=self.sigma_dyn , frame=self.frame, template=self.template )
        self.update(self.particles, self.weights, frame, sigma_exp=self.sigma_exp, template=self.template )
         
        temp_center = self.particles[np.argmax(self.weights ), :]
        temp_template = frame[(int(temp_center[0])-self.template.shape[0]/2) : (int(temp_center[0])-self.template.shape[0]/2+self.template.shape[0])
                             ,(int(temp_center[1])-self.template.shape[1]/2) : (int(temp_center[1])-self.template.shape[1]/2+self.template.shape[1])]
        
        self.template = self.alpha*temp_template + (1-self.alpha) *self.template
#        
#        temp_center = particles[np.argmax(weights ), :]
#        temp_template = frame[(int(temp_center[0])-template.shape[0]/2) : (int(temp_center[0])-template.shape[0]/2+template.shape[0])
#                        ,(int(temp_center[1])-template.shape[1]/2) : (int(temp_center[1])-template.shape[1]/2+template.shape[1])]
#        
#        plt.imshow(temp_template)
#        plt.imshow( template)
#        alpha = .8
#        plt.imshow(alpha*temp_template + (1-alpha) *template)
        
        
        
#         RENDER METHOD
#        u_weighted_mean = 0
#        v_weighted_mean = 0
#        for i in range(len(self.particles)):
#            u_weighted_mean +=  self.particles[i, 0] *  self.weights[i]
#            v_weighted_mean +=  self.particles[i, 1] *  self.weights[i]
#        self.template = frame[(int(u_weighted_mean)-self.template.shape[0]/2) : (int(u_weighted_mean)-self.template.shape[0]/2+self.template.shape[0])
#                                ,(int(v_weighted_mean)-self.template.shape[1]/2) : (int(v_weighted_mean)-self.template.shape[1]/2+self.template.shape[1])
#                                ]
#    
#        temp_template = frame[(x0-template.shape[0]/2) : (x0-template.shape[0]/2+template.shape[0])
#                             ,(y0-template.shape[1]/2) : (y0-template.shape[1]/2+template.shape[1])]

#cv2.circle is (col,row)
#u=y=row, v=x=col
#when working with particles
#when drawing, flip it
#so the way I use when getting coordinates from particle is cy,cx = particles[i]
#from then on, I calculate other coordinates, say x1,y1, x2,y2...
#for drawing, just use x,y as calculated
#hope that makes sense

         
        self.resample()


class MeanShiftLitePF(ParticleFilter):
    """A variation of particle filter tracker that uses the color distribution of the patch."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the Mean Shift Lite particle filter object (parameters are the same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above. There is one element that is added
        called alpha which is explained in the problem set documentation. By calling super(...) all the elements used
        in ParticleFilter will be inherited so you don't have to declare them again."""

        super(MeanShiftLitePF, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.num_bins = kwargs.get('hist_bins_num', 8)  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
    def update_ms(self,particles, weights, frame, sigma_exp, template):

        distance = []
        frame_green =frame[:,:,1].astype(float) 
        template_green = template[:,:,1].astype(float)
        for k in range(len(particles)):
            lx = int( particles[k,0]-template.shape[0]/2)
            ux = int( particles[k,0]-template.shape[0]/2)+ template.shape[0]  
            ly = int( particles[k,1]-template.shape[1]/2)
            uy = int( particles[k,1]-template.shape[1]/2)+ template.shape[1] 
            pf = frame [ lx:ux
                         ,ly:uy    ]
            
            pf0 = cv2.calcHist([pf],[0],None,[8],[0,256])
            pf1 = cv2.calcHist([pf],[1],None,[8],[0,256])
            pf2 = cv2.calcHist([pf],[2],None,[8],[0,256])
            
            tp0 = cv2.calcHist([template],[0],None,[8],[0,256])
            tp1 = cv2.calcHist([template],[1],None,[8],[0,256])
            tp2 = cv2.calcHist([template],[2],None,[8],[0,256])
        
            
            pf_all = np.concatenate((pf0,pf1,pf2), 0)
            tp_all = np.concatenate((tp0,tp1,tp2), 0)
            
            num = (pf_all - tp_all) **2
            den = pf_all + tp_all
            
            
            with np.errstate(divide='ignore'):
                d = num / den
                d[den == 0] = 0
                
            distance.append(np.sum(d))
            
        p = np.exp(-np.array(distance)/2.0/(sigma_exp**2)) 
#        print 'min dist', particles[ int(np.argmax(np.sum(d))),:], distance[int(np.sum(d))]
        self.weights = p/sum(p) 
        
#        k = np.argmin(distance)  #np.argmax(p/sum(p))
#        print k
#        lx = int( particles[k,0]-template.shape[0]/2)
#        ux = int( particles[k,0]-template.shape[0]/2+ template.shape[0])  
#        ly = int( particles[k,1]-template.shape[1]/2)
#        uy = int( particles[k,1]-template.shape[1]/2+ template.shape[1])
#        pf = frame_green[ lx:ux,ly:uy    ]
#        plt.imshow(pf)
#        plt.imshow(frame)
#        plt.imshow(template_grey)
#        print 'dist', np.linalg.norm( pf  - template_grey )  # 4291.93126692
#        
        
    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "Mean Shift Lite" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        
        self.predict(self.particles, sigma_dyn=self.sigma_dyn , frame=self.frame, template=self.template )
        self.update_ms(self.particles, self.weights, frame, sigma_exp=self.sigma_exp, template=self.template )
        self.resample(  )


class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object (parameters same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        pass
