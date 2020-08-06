# Solve numerically the system described by Jen.
# The PDEs are discretised spatially and the resulting ODE
# system is solved using the built-in bdf method in scipy.

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import simps, LSODA, BDF, RK45
from timeit import default_timer as timer
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from random import seed
from random import shuffle

# Create class to do dynamic printing on one line

class Printer():
    """Print things to stdout on one line dynamically"""
    def __init__(self,data):
        sys.stdout.write("\r\x1b[K"+data.__str__())
        sys.stdout.flush()

#####################################

def plot_all_times_sca():
    
    length = 200.0
    dx = 0.1
    Nx = int(length/dx) + 1
    x = np.linspace(0,length,Nx)
        
    # Initialise condition

    Y_0 = np.zeros(Nx)
    
    Y_0[np.int((Nx-1)/2 - 25/dx):np.int((Nx-1)/2+ 25/dx)] = 0.5
    
    Y_init=Y_0
        
    start_times = [0,10,100]
    end_times = [10,100,1000]
    
    solutions = [np.zeros(2*Nx) for ii in range(3)]
    
    list_s=[0.8,0.5,0.3]

    for s in list_s:
            
        Y_0=Y_init
            
        for ii, (t_start,t_end) in enumerate(zip(start_times, end_times)):
        
            solutions[ii] = sca(Y_0, t_start, s, run_length_max = t_end, plotting = False)
            Y_0 = solutions[ii]
        
        fig, ax = plt.subplots(figsize=[8,5])
        ax.set_title('Single cell \n where $s$=' + str(s))    
        ax.plot(x,Y_init,'b--', label='$M_{init}$')
    
        for ii in range(2):
            ax.plot(x,solutions[ii],'b')
    
        ax.plot(x,solutions[2],'b', label='$M$')
    
        ax.legend()
        fig.savefig('Figures/sca_s_0'+str(s)[2]+'.png')
        plt.close('all')
            
        solutions=[Y_init]+solutions
        np.save('Figures/sca_s_0'+str(s)[2],solutions)

def sca(M_0, t_0, s, run_length_max = 1000, plotting = False):
    
    ######################### Initial setup ##########################
        
    # Space discretisation
    
    length = 200.
    dx = 0.1   
    Nx = int(length/dx) + 1
    x = np.linspace(0,length,Nx)
    
    # Intermediate constants
    
    P_m = 1.0
    r = 1 - s
    
    # Initialise condition

    #t_0 = 0.0
    #M_0 = np.exp(-0.01*(x - length/2)**2)
    
    #M_0 = np.zeros(Nx)
    #M_0[np.int((Nx-1)/2 - 25/dx):np.int((Nx-1)/2 + 25/dx)] = 1.0
    
    # Initialise figure if plotting

    if plotting:
        fig, ax = plt.subplots(figsize=[8,5])
        
        ax.plot(x,M_0)
        ax.set_title('$M$')
        
        plt.pause(0.01)
    
    ########################## Build ODE RHS ###########################

    def f(t, M):
        
        # Initialise output
        
        dMdt = np.zeros(Nx)
        
        # Manipulate M to get M_j-1 etc. - assuming periodic BCs
        
        M_jplus2 = np.roll(M, -2) # Moving elements back two means we are now indexing the j+2'th element in the j'th place
        
        M_jplus1 = np.roll(M, -1) # Moving elements back one means we are now indexing the j+1'th element in the j'th place
        
        M_jminus1 = np.roll(M, 1) # Moving elements forward one means we are now indexing the j-1'th element in the j'th place
        
        M_jminus2 = np.roll(M, 2) # Moving elements forward two means we are now indexing the j-2'th element in the j'th place
        
        # Calculate RHS
        
        dMdx = (M_jplus2 - M_jminus2)/(4*dx)  # fourth order
        
        d2Mdx2 = (-M_jplus2 + 16*M_jplus1 - 30*M + 16*M_jminus1 - M_jminus2)/(12*dx*dx) # fourth order
        
        dMdt = P_m*((s + (r - s)*M*(3*M - 2))*d2Mdx2
                + (r - s)*(6*M - 2)*dMdx**2)/2.0
        
        #dMdt = P_m*((r + (s - r)*(1 + 5*M)*(1 - M))*d2Mdx2
        #        + 2*(s - r)*(2 - 5*M)*dMdx**2)/2.0
        
        return dMdt
    
    ############################# Solve ODE ############################
    
    # Set up ODE integrator
    
    solver = LSODA(f, t_0, M_0, run_length_max)
    
    # Integrate ODE forward
    
    while solver.t < run_length_max:
        
        # Step solver forward
        
        solver.step()
        
        output = 'Run time = %.6f s.' % solver.t
        
        Printer(output)
        
        # If plotting is enabled do plotting every timestep
        
        if plotting:
            
            # Plot title
            
            fig.suptitle('Model at ' + str('{:.2f}'.format(solver.t)) + 's.')
            
            # Plot concentrations on 5 different axes
            
            ax.clear()
            ax.plot(x,M_0)
            ax.plot(x,solver.y)
            ax.set_title('$M$')
            
            plt.pause(0.01)
    
    ############################## Output ##############################
    
    # Interactive output
    
    M = solver.y
    
    return M
    
def stoch_simulation(rho, run_length_max = 1000, plotting = False):
    
    
    n = 5
    from_list = mpl.colors.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.Set1(range(2,4)))
    
    #top=cm.get_cmap('Yellows',100)
    #bottom=cm.get_cmap('Blues',100)
    #newcolors= np.vstack((top(np.linspace(0,1,100)),bottom((np.linspace(0,1,100)))))
    #newcmp=ListedColormap(newcolors, name='orangeblue')
    
    length=30
    width=30
    
    motility_ratem=1.0
    motility_ratex=1.0
    
    #Initial condition       
    
    domain_matrix_0=np.random.uniform(size=(width,length))
    #domain_matrix_0[domain_matrix_0>=0.9]=0
    domain_matrix_0[(domain_matrix_0>=0.5)*(domain_matrix_0<=1)]=1
    domain_matrix_0[(domain_matrix_0<0.5)*(domain_matrix_0>0)]=2

    j=0
    rec_time=100
    solution=np.zeros((width, length, (run_length_max//rec_time)+1))
    
    #domain_matrix_0 = np.ones((width, length))
    #domain_matrix_0[:,np.int((length-1)/2)-25:np.int((length-1)/2)] = np.random.uniform(size=(width,25))
    #domain_matrix_0[domain_matrix_0>0.5]=1

    #domain_matrix_0[:,np.int((length-1)/2):np.int((length-1)/2)+25] = np.random.uniform(size=(width,25))
    #domain_matrix_0[domain_matrix_0<0.5]=0
    #domain_matrix_0[(domain_matrix_0>0.5)*(domain_matrix_0<1)]=2
    
    
    M_0=np.mean(domain_matrix_0==1,axis=0)
    X_0=np.mean(domain_matrix_0==2,axis=0)
    
    countm=np.sum(domain_matrix_0==1)
    countx=np.sum(domain_matrix_0==2)

    tau_sum=run_length_max
    domain_matrix=domain_matrix_0
    x = np.arange(200)
    t=0
    # Initialise figure if plotting

    if plotting:
        fig = plt.figure()
        ax2=fig.add_subplot        
        plt.pause(0.01)
        plt.axis('off')
        
    plot_now=0
    
    while t < tau_sum:
    
        #Plotting during
        if plotting and (t>=plot_now):
            
            #fig.suptitle('Model at ' + str('{:.2f}'.format(t)) + 's.')
            ax2=fig.add_subplot()
            pos = ax2.imshow(domain_matrix,cmap=cm) 
            
            solution[:,:,j]=domain_matrix
            j+=1
            plt.pause(0.01)
            plot_now+=rec_time

        #Gillespie algorithm
        a0 =  (motility_ratem*countm+motility_ratex*countx);
        tau  = (1/a0)*np.log(1/(np.random.uniform()));
        t = t + tau;
    
        eventchooser = np.random.uniform()
    
        R = np.int(np.random.uniform()*width)
        C = np.int(np.random.uniform()*length)
    
        ##Decide which cell will move based on motility rates
        if (eventchooser < (motility_ratem*countm)/(a0)):
            target_element=1
            #Element is attracted/repelled by
            choose_element=1
            #Other element is attracted/repelled by 
            choose_element_2=2 
        else:
            target_element=2
            choose_element=2 
            choose_element_2=1 
        
    
        while (domain_matrix[R,C] != target_element):
            R = np.int(np.random.uniform()*width)
            C = np.int(np.random.uniform()*length)
         
         ## SELECT MOVEMENT (1D)
        movement_selector = np.random.uniform()
        swap_selector=np.random.uniform()
        adhesive_prob=np.random.uniform()
        move=0;
        
        #Homotypic interaction
        
        #periodic boundary conditions - decides whether the element can move based on adhesive properties
        #if at least one is occupied in ball
                    
        #innocent until proven otherwise
        make_bonds=False
        break_bonds=False
        swap_make_bonds=False
        swap_break_bonds=False
    
        if movement_selector<0.25:
            pos=1
            if (domain_matrix[R,np.int(min(np.mod(C+2,length),C+2))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(min(np.mod(C+2,length),C+2))]==choose_element or domain_matrix[np.int(max(np.mod(R-1,length),R-1)),np.int(min(np.mod(C+2,length),C+2))]==choose_element):
                make_bonds=True
            if  (domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(max(np.mod(R-1,length),R-1)),np.int(min(np.mod(C-1,length),C-1))]==choose_element):
                break_bonds=True
            if (domain_matrix[R,np.int(min(np.mod(C+2,length),C+2))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(min(np.mod(C+2,length),C+2))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,length),R-1)),np.int(min(np.mod(C+2,length),C+2))]==choose_element_2):
                swap_break_bonds=True
            if  (domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,length),R-1)),np.int(min(np.mod(C-1,length),C-1))]==choose_element_2):
                swap_make_bonds=True
        elif movement_selector<0.5:
            pos=2
            if (domain_matrix[R,np.int(max(np.mod(C-2,length),C-2))]==choose_element or domain_matrix[np.int(max(np.mod(R-1,length),R-1)),np.int(max(np.mod(C-2,length),C-2))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(max(np.mod(C-2,length),C-2))]==choose_element):
                make_bonds=True
            if (domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element or domain_matrix[np.int(min(np.mod(R-1,length),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element):
                break_bonds=True
            if (domain_matrix[R,np.int(max(np.mod(C-2,length),C-2))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,length),R-1)),np.int(max(np.mod(C-2,length),C-2))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(max(np.mod(C-2,length),C-2))]==choose_element_2):
                swap_break_bonds=True
            if (domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R-1,length),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,length),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2):
                swap_make_bonds=True
        elif movement_selector<0.75: 
            pos=3
            if (domain_matrix[np.int(min(np.mod(R+2,width),R+2)),C]==choose_element or domain_matrix[np.int(min(np.mod(R+2,width),R+2)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(min(np.mod(R+2,width),R+2)),np.int(min(np.mod(C+1,length),C+1))]==choose_element):
                make_bonds=True
            if (domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]==choose_element or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element):
                break_bonds=True
            if (domain_matrix[np.int(min(np.mod(R+2,width),R+2)),C]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+2,width),R+2)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+2,width),R+2)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2):
                swap_break_bonds=True
            if (domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2):
                swap_make_bonds=True
        else: 
            pos=4
            if (domain_matrix[np.int(max(np.mod(R-2,width),R-2)),C]==choose_element or domain_matrix[np.int(max(np.mod(R-2,width),R-2)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(max(np.mod(R-2,width),R-2)),np.int(min(np.mod(C+1,length),C+1))]==choose_element): 
                make_bonds=True
            if (domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element): 
                break_bonds=True
            if (domain_matrix[np.int(max(np.mod(R-2,width),R-2)),C]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-2,width),R-2)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-2,width),R-2)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2): 
                swap_break_bonds=True
            if (domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2): 
                swap_make_bonds=True

        move=False
        swap=False
        
        
        if target_element==1: 
        #For M
        #For adhesion
        #Probability of movement given that you will make bonds and break bonds
            u=0.2
        #Probability of movement given that you will make bonds and not break bonds
            p=1.0
        #Probability of movement given that you will break bonds and not make bonds
            q=0.0
        #Probability of movement given that you will not make or break bonds
            s=1.0
        elif target_element==2:
        #For repulsion of other
        #Probability of movement given that you will make bonds and break bonds
            u=0.2
        #Probability of movement given that you will make bonds and not break bonds
            p=1.0
        #Probability of movement given that you will break bonds and not make bonds
            q=0.0
        #Probability of movement given that you will not make or break bonds
            s=1.0           
        
        #Now decide if move is acceptable
        if make_bonds and break_bonds and np.random.uniform()<u:
            move=True
        elif make_bonds and break_bonds==False and np.random.uniform()<p:
            move=True
        elif make_bonds==False and break_bonds and np.random.uniform()<q:
            move=True
        elif make_bonds==False and break_bonds==False and np.random.uniform()<s:
            move=True
        
        #Decide if a swap is acceptable (consider the other type)
        if swap_selector<rho:
            if choose_element==1: 
        #For M
        #For adhesion
        #Probability of movement given that you will make bonds and break bonds
                u=0.2
        #Probability of movement given that you will make bonds and not break bonds
                p=1.0
        #Probability of movement given that you will break bonds and not make bonds
                q=0.0
        #Probability of movement given that you will not make or break bonds
                s=1.0
            elif choose_element==2:
        #For repulsion of other
        #Probability of movement given that you will make bonds and break bonds
                u=0.2
        #Probability of movement given that you will make bonds and not break bonds
                p=1.0
        #Probability of movement given that you will break bonds and not make bonds
                q=0.0
        #Probability of movement given that you will not make or break bonds
                s=1.0   
        
        #Now decide if swap is acceptable
            if swap_make_bonds and swap_break_bonds and np.random.uniform()<u:
                swap=True
            elif swap_make_bonds and swap_break_bonds==False and np.random.uniform()<p:
                swap=True
            elif swap_make_bonds==False and swap_break_bonds and np.random.uniform()<q:
                swap=True
            elif swap_make_bonds==False and swap_break_bonds==False and np.random.uniform()<s:
                swap=True
        
        if move:
            if pos==1:
                if domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))] == 0:
                    domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))] = domain_matrix[R,C]
                    domain_matrix[R,C] = 0
                elif swap and swap_selector<rho:
                    store=domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]
                    domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))] = domain_matrix[R,C]
                    domain_matrix[R,C] = store
            elif pos==2:
                if domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))] == 0:
                    domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]  = domain_matrix[R,C]
                    domain_matrix[R,C] = 0
                elif swap and swap_selector<rho:
                    store=domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]
                    domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]  = domain_matrix[R,C]
                    domain_matrix[R,C] = store 
            elif pos==3:
                if domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C] == 0:
                    domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = 0
                elif swap and swap_selector<rho:
                    store=domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]
                    domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = store      
            elif pos==4:
                if domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C] == 0:
                    domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = 0  
                elif swap and swap_selector<rho:
                    store=domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]
                    domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = store      
                                
    #np.save('Figures/Moore_u_'+str(u)[0]+str(u)[2]+'_p_'+str(p)[0]+str(p)[2]+'_q_'+str(q)[0]+str(q)[2]+'_s_'+str(q)[0]+str(q)[2]+'_rho_'+str(rho)[0]+str(rho)[2], solution)
    fig.savefig('Figures/Moore_u_'+str(u)[0]+str(u)[2]+'_p_'+str(p)[0]+str(p)[2]+'_q_'+str(q)[0]+str(q)[2]+'_s_'+str(q)[0]+str(q)[2]+'_rho_'+str(rho)[0]+str(rho)[2]+'.png')
    return domain_matrix, solution

    
def stoch_simulation_von_neumann(rho, run_length_max = 1000, plotting = False):
    
    n = 5
    from_list = mpl.colors.LinearSegmentedColormap.from_list
    newcmp = from_list(None, plt.cm.Set1(range(2,4)))
    
    length=50
    width=1
    
    motility_ratem=1.0
    motility_ratex=1.0
    
    #Initial condition       
    domain_matrix_0=np.random.uniform(size=(width,length))
    
    sequence = [i for i in range(length)]
    # randomly shuffle the sequence
    shuffle(sequence)
    sequence=np.array([sequence])
    sequence[(sequence<25)]=1
    sequence[(sequence>=25)]=2
    domain_matrix_0=sequence
    
    #import pdb; pdb.set_trace()
    #domain_matrix_0=np.random.uniform(size=(width,length))
    #domain_matrix_0[domain_matrix_0>=0.9]=0
    #domain_matrix_0[(domain_matrix_0>=0.5)*(domain_matrix_0<=1)]=1
    #domain_matrix_0[(domain_matrix_0<0.5)*(domain_matrix_0>0)]=2

    j=0
    rec_time=1000
    solution=np.zeros((width, length, (run_length_max//rec_time)+1))
    
    #domain_matrix_0 = np.ones((width, length))
    #domain_matrix_0[:,np.int((length-1)/2)-25:np.int((length-1)/2)] = np.random.uniform(size=(width,25))
    #domain_matrix_0[domain_matrix_0>0.5]=1

    #domain_matrix_0[:,np.int((length-1)/2):np.int((length-1)/2)+25] = np.random.uniform(size=(width,25))
    #domain_matrix_0[domain_matrix_0<0.5]=0
    #domain_matrix_0[(domain_matrix_0>0.5)*(domain_matrix_0<1)]=2
    
    
    M_0=np.mean(domain_matrix_0==1,axis=0)
    X_0=np.mean(domain_matrix_0==2,axis=0)
    
    countm=np.sum(domain_matrix_0==1)
    countx=np.sum(domain_matrix_0==2)

    tau_sum=run_length_max
    domain_matrix=domain_matrix_0
    x = np.arange(200)
    t=0
    # Initialise figure if plotting

    if plotting:
        fig = plt.figure()
        ax2=fig.add_subplot        
        plt.pause(0.01)
        plt.axis('off')
        
    plot_now=0
    
    while t < tau_sum:
    
        #Plotting during
        if plotting and (t>=plot_now):
            
            #fig.suptitle('Model at ' + str('{:.2f}'.format(t)) + 's.')
            ax2=fig.add_subplot()
            pos = ax2.imshow(domain_matrix,cmap=newcmp) 
            
            solution[:,:,j]=domain_matrix
            j+=1
            plt.pause(0.01)
            plot_now+=rec_time

        #Gillespie algorithm
        a0 =  (motility_ratem*countm+motility_ratex*countx);
        tau  = (1/a0)*np.log(1/(np.random.uniform()));
        t = t + tau;
    
        eventchooser = np.random.uniform()
    
        R = np.int(np.random.uniform()*width)
        C = np.int(np.random.uniform()*length)
    
        ##Decide which cell will move based on motility rates
        if (eventchooser < (motility_ratem*countm)/(a0)):
            target_element=1
            #Element is attracted/repelled by
            choose_element=1
            #Other element is attracted/repelled by 
            choose_element_2=2 
        else:
            target_element=2
            choose_element=2
            choose_element_2=1 
        
    
        while (domain_matrix[R,C] != target_element):
            R = np.int(np.random.uniform()*width)
            C = np.int(np.random.uniform()*length)
         
         ## SELECT MOVEMENT (1D)
        movement_selector = np.random.uniform()
        swap_selector=np.random.uniform()
        adhesive_prob=np.random.uniform()
        move=0;
        
        #Homotypic interaction
        
        #periodic boundary conditions - decides whether the element can move based on adhesive properties
        #if at least one is occupied in ball
                    
        #innocent until proven otherwise
        make_bonds=False
        break_bonds=False
        make_bonds_with_edge=False
        break_bonds_with_edge=False
        swap_make_bonds=False
        swap_break_bonds=False
        swap_make_bonds_with_edge=False
        swap_break_bonds_with_edge=False
    
        if movement_selector<0.5:
            pos=1
            if (domain_matrix[R,np.int(min(np.mod(C+2,length),C+2))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element):
                make_bonds=True
            if  (domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]==choose_element):
                break_bonds=True
            if np.int(np.mod(C+1,length-1))==0 or np.mod(R,length-1)==0:
                make_bonds_with_edge=True
                
            swapping_element=domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))] #implies that it equals the other element
            if swapping_element!=target_element:
                if (domain_matrix[R,np.int(min(np.mod(C+2,length),C+2))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2):
                    swap_break_bonds=True
                if  (domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]==choose_element_2):
                    swap_make_bonds=True
                if np.int(np.mod(C+1,length-1))==0 or np.mod(R,length-1)==0:
                    swap_break_bonds_with_edge=True
                    
        elif movement_selector>0.5:
            pos=2
            if (domain_matrix[R,np.int(max(np.mod(C-2,length),C-2))]==choose_element or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element):
                make_bonds=True
            if (domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element or domain_matrix[np.int(min(np.mod(R-1,width),R-1)),C]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element):
                break_bonds=True
            if np.int(np.mod(C-1,length-1))==0 or np.mod(R,length-1)==0:
                make_bonds_with_edge=True   
                            
            swapping_element=domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))] #implies that it equals the other element
            if swapping_element!=target_element: 
                if (domain_matrix[R,np.int(max(np.mod(C-2,length),C-2))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2):
                    swap_break_bonds=True
                if (domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R-1,width),R-1)),C]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element_2):
                    swap_make_bonds=True
                if np.int(np.mod(C-1,length-1))==0 or np.mod(R,length-1)==0:
                    swap_break_bonds_with_edge=True
                    
        elif movement_selector>2: 
            pos=3
            if (domain_matrix[np.int(min(np.mod(R+2,width),R+2)),C]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element):
                make_bonds=True
            if (domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]==choose_element or domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element):
                break_bonds=True
            if np.int(np.mod(R+1,length-1))==0 or np.mod(C,length-1)==0:
                make_bonds_with_edge=True
                
            swapping_element=domain_matrix[np.int(min(np.mod(R+1,length),R+1)),C] #implies that it equals the other element
            if swapping_element!=target_element:
                if (domain_matrix[np.int(min(np.mod(R+2,width),R+2)),C]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2):
                    swap_break_bonds=True
                if (domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]==choose_element_2 or domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element_2):
                    swap_make_bonds=True
                if np.int(np.mod(R+1,length-1))==0 or np.mod(C,length-1)==0:
                    make_bonds_with_edge=True
                    swap_break_bonds_with_edge=True
        elif movement_selector>2: 
            pos=4
            if (domain_matrix[np.int(max(np.mod(R-2,width),R-2)),C]==choose_element or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element): 
                make_bonds=True
            if (domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element or domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]==choose_element or domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element): 
                break_bonds=True
            if np.int(np.mod(R-1,length-1))==0 or np.mod(C,length-1)==0:
                make_bonds_with_edge=True   
                        
            swapping_element=domain_matrix[np.int(max(np.mod(R-1,length),R-1)),C] #implies that it equals the other element
            if swapping_element!=target_element:
                if (domain_matrix[np.int(max(np.mod(R-2,width),R-2)),C]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[np.int(max(np.mod(R-1,width),R-1)),np.int(min(np.mod(C+1,length),C+1))]==choose_element_2): 
                    swap_break_bonds=True
                if (domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]==choose_element_2 or domain_matrix[np.int(min(np.mod(R+1,width),R+1)),np.int(max(np.mod(C-1,length),C-1))]==choose_element_2 or domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]==choose_element_2): 
                    swap_make_bonds=True
                if np.int(np.mod(R-1,length-1))==0 or np.mod(C,length-1)==0:
                    swap_break_bonds_with_edge=True

        if  np.mod(C,length-1)==0 or np.mod(R,length-1)==0:
            break_bonds_with_edge=True
            swap_make_bonds_with_edge=True
            
        move=False
        swap=False
        
        
        if target_element==1: 
        #For M
        #For adhesion
        #Probability of movement given that you will make bonds and break bonds
            u=0.2 #1
        #Probability of movement given that you will make bonds and not break bonds
            p=0.0
        #Probability of movement given that you will break bonds and not make bonds
            q=1.0 #0.1
        #Probability of movement given that you will not make or break bonds
            s=1.0
            if make_bonds and break_bonds and np.random.uniform()<u:
                move=True
            elif make_bonds and break_bonds==False and np.random.uniform()<p:
                move=True
            elif make_bonds==False and break_bonds and np.random.uniform()<q:
                move=True
            elif make_bonds==False and break_bonds==False and np.random.uniform()<s:
                move=True
        elif target_element==2:
        #For repulsion of other
        #Probability of movement given that you will make bonds and break bonds
            u=0.2 #0.5
        #Probability of movement given that you will make bonds and not break bonds
            p=0.0
        #Probability of movement given that you will break bonds and not make bonds
            q=1.0
        #Probability of movement given that you will not make or break bonds
            s=1.0
                
            #if make_bonds_with_edge and break_bonds_with_edge and np.random.uniform()<u: #1
            #   move=True
            #elif make_bonds_with_edge and break_bonds_with_edge==False and np.random.uniform()<p:
            #   move=True
            #elif make_bonds_with_edge==False and break_bonds_with_edge and np.random.uniform()<q:
            #   move=True
            #elif make_bonds_with_edge==False and break_bonds_with_edge==False and np.random.uniform()<s:
            #   move=True
            
            
            if make_bonds and break_bonds and np.random.uniform()<u: #1
                swap=True
            elif make_bonds and break_bonds==False and np.random.uniform()<p: #1
                swap=True
            elif make_bonds==False and break_bonds and np.random.uniform()<q: #0
                swap=True
            elif make_bonds==False and break_bonds==False and np.random.uniform()<s: #0.1
                swap=True
                
#               if make_bonds and break_bonds and np.random.uniform()<1:
#                   move=True
#               elif make_bonds and break_bonds==False and np.random.uniform()<1:
#                   move=True
#               elif make_bonds==False and break_bonds and np.random.uniform()<0:
#                   move=True
#               elif make_bonds==False and break_bonds==False and np.random.uniform()<0.1:
#                   move=True   
        
        #Now decide if move is acceptable
        

        
        #Decide if a swap is acceptable (consider the other type) #no reason to swap if its with the same cell type
        if swap_selector<rho and swapping_element!=target_element: 
            if swapping_element==1: 
        #For M
        #For adhesion
        #Probability of movement given that you will make bonds and break bonds
                u=0.2 #1
        #Probability of movement given that you will make bonds and not break bonds
                p=0.0
        #Probability of movement given that you will break bonds and not make bonds
                q=1.0
        #Probability of movement given that you will not make or break bonds
                s=1.0
                #Now decide if swap is acceptable
                if swap_make_bonds and swap_break_bonds and np.random.uniform()<u:
                    swap=True
                elif swap_make_bonds and swap_break_bonds==False and np.random.uniform()<p:
                    swap=True
                elif swap_make_bonds==False and swap_break_bonds and np.random.uniform()<q:
                    swap=True
                elif swap_make_bonds==False and swap_break_bonds==False and np.random.uniform()<s:
                    swap=True
            elif swapping_element==2:
        #For repulsion of other
        #Probability of movement given that you will make bonds and break bonds
                u=0.2 #0.5
        #Probability of movement given that you will make bonds and not break bonds
                p=0.0
        #Probability of movement given that you will break bonds and not make bonds
                q=1.0
        #Probability of movement given that you will not make or break bonds
                s=1.0
                #if swap_make_bonds_with_edge and swap_break_bonds_with_edge and np.random.uniform()<u: #1
                #   swap=True
                #elif swap_make_bonds_with_edge and swap_break_bonds_with_edge==False and np.random.uniform()<p:
                #   swap=True
                #elif swap_make_bonds_with_edge==False and swap_break_bonds_with_edge and np.random.uniform()<q:
                #   swap=True
                #elif swap_make_bonds_with_edge==False and swap_break_bonds_with_edge==False and np.random.uniform()<s:
                #   swap=True
                    
                    
                if swap_make_bonds and swap_break_bonds and np.random.uniform()<u: #1
                    swap=True
                elif swap_make_bonds and swap_break_bonds==False and np.random.uniform()<p: #1
                    swap=True
                elif swap_make_bonds==False and swap_break_bonds and np.random.uniform()<q: #0
                    swap=True
                elif swap_make_bonds==False and swap_break_bonds==False and np.random.uniform()<s: #0.1
                    swap=True

        
        if move:
            if pos==1:
                if domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))] == 0:
                    domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))] = domain_matrix[R,C]
                    domain_matrix[R,C] = 0
                elif swap and swap_selector<rho:
                    store=domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))]
                    domain_matrix[R,np.int(min(np.mod(C+1,length),C+1))] = domain_matrix[R,C]
                    domain_matrix[R,C] = store
            elif pos==2:
                if domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))] == 0:
                    domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]  = domain_matrix[R,C]
                    domain_matrix[R,C] = 0
                elif swap and swap_selector<rho:
                    store=domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]
                    domain_matrix[R,np.int(max(np.mod(C-1,length),C-1))]  = domain_matrix[R,C]
                    domain_matrix[R,C] = store 
            elif pos==3:
                if domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C] == 0:
                    domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = 0
                elif swap and swap_selector<rho:
                    store=domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C]
                    domain_matrix[np.int(min(np.mod(R+1,width),R+1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = store      
            elif pos==4:
                if domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C] == 0:
                    domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = 0  
                elif swap and swap_selector<rho:
                    store=domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C]
                    domain_matrix[np.int(max(np.mod(R-1,width),R-1)),C] = domain_matrix[R,C]
                    domain_matrix[R,C] = store      
                                
    np.save('Stoch_solutions', solution)
    fig.savefig('Figures/VN_1D_u_'+str(u)[0]+str(u)[2]+'_p_'+str(p)[0]+str(p)[2]+'_q_'+str(q)[0]+str(q)[2]+'_s_'+str(q)[0]+str(q)[2]+'_rho_'+str(rho)[0]+str(rho)[2]+'.png',bbox_inches = 'tight')

    return domain_matrix, solution

        
def animate_for_me(soln):
    
    from matplotlib import animation
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    t=0.0
    fig.suptitle('Model at ' + str('{:.2f}'.format(t)) + 's.')
    pos = ax.imshow(soln[:,:,0],cmap=plt.cm.get_cmap('Blues', 3))        

    def animate(i, soln, plot):
        
        ax.clear()
        t=10*i
        fig.suptitle('Model at ' + str('{:.2f}'.format(t)) + 's.')
        plot = ax.imshow(soln[:,:,i],cmap=plt.cm.get_cmap('Blues', 3))

        return plot,
    
    anim = animation.FuncAnimation(fig, animate, fargs=(soln, pos), frames = soln.shape[-1] - 1, interval = 10)
    
    writer = animation.FFMpegWriter(bitrate=5000, fps=5)
    anim.save('anim_trial_run.mp4', dpi=300, writer=writer)

def plot_all_times():
    
    length = 200.0
    dx = 0.1
    Nx = int(length/dx) + 1
    x = np.linspace(0,length,Nx)
        
    # Initialise condition

    #M_0 = 0.5*np.ones(Nx)
    #X_0 = 0.5*np.ones(Nx)
    M_0 = 0.5*np.zeros(Nx)
    X_0 = 0.5*np.zeros(Nx)
    
    
    M_0[np.int((Nx-1)/2 - 25/dx):np.int((Nx-1)/2)] = 0.5
    X_0[np.int((Nx-1)/2):np.int((Nx-1)/2 + 25/dx)] = 0.5
    
    M_init=M_0
    X_init=X_0
        
    start_times = [0,10,100]
    end_times = [10,100,1000]
    
    solutions = [np.zeros(2*Nx) for ii in range(3)]
    
    list_rho=[1.0]
    list_s_x=[0.8,0.3]
    list_s_m=[0.8,0.3]
    
    for rho in list_rho:
        for s_M in list_s_m:
            for s_X in list_s_x:
                
                M_0=M_init
                X_0=X_init
            
                for ii, (t_start,t_end) in enumerate(zip(start_times, end_times)):
        
                    solutions[ii] = dca(rho, M_0, X_0, t_start, s_M, s_X, run_length_max = t_end, plotting = False)
                    M_0 = solutions[ii][:Nx]
                    X_0 = solutions[ii][Nx:] 
        
                fig, ax = plt.subplots(figsize=[8,5])
    
                ax.set_title('Homotypic interaction \n where $s_M$=' + str(s_M) +' and $s_X$=' + str(s_X))    
                ax.plot(x,M_init,'b--', label='$M_{init}$')
                ax.plot(x,X_init,'g--', label='$X_{init}$')
    
                for ii in range(2):
                    M=solutions[ii][:Nx]
                    X=solutions[ii][Nx:]
                    ax.plot(x,M,'b')
                    ax.plot(x,X,'g')
    
                M=solutions[2][:Nx]
                X=solutions[2][Nx:]
                ax.plot(x,M,'b', label='$M$')
                ax.plot(x,X,'g', label='$X$')
    
                ax.legend()
                fig.savefig('Figures/Hom_swap_6_rho_'+str(rho)[0]+str(rho)[2]+'_s_M_0'+str(s_M)[2]+'_s_X_0'+str(s_X)[2]+'.png')
                plt.close('all')
            
                Y_0=np.concatenate((M_init,X_init))
                solutions=[Y_0]+solutions
                np.save('Figures/Hom_swap_6_rho_'+str(rho)[0]+str(rho)[2]+'_s_M_0'+str(s_M)[2]+'_s_X_0'+str(s_X)[2],solutions)
    
def dca(rho, M_0, X_0, t_0, s_M, s_X, run_length_max = 1000, plotting = False):
    
    ######################### Initial setup ##########################
        
    # Space discretisation
    
    length = 200.
    dx = 0.1   
    Nx = int(length/dx) + 1
    x = np.linspace(0,length,Nx)
    
    # Intermediate constants
    
    P_m = 1.0
    r_M = 1 - s_M
    r_X = 1 - s_X
    
    # Initialise condition

    #t_0 = 0.0
    #M_0 = np.exp(-0.01*(x - length/2)**2)
    
    #X_0 = np.zeros(Nx)
    
    #M_0[np.int((Nx-1)/2 - 25/dx):np.int((Nx-1)/2)] = 0.5
    #X_0[np.int((Nx-1)/2):np.int((Nx-1)/2 + 25/dx)] = 0.5
    
    Y_0=np.concatenate((M_0,X_0))
    init_conc_M0=simps(M_0,x)
    init_conc_X0=simps(X_0,x)
    
    # Initialise figure if plotting

    if plotting:
        fig, ax = plt.subplots(figsize=[8,5])
        
        ax.plot(x,M_0,'b--')
        ax.plot(x,X_0,'g--')
        
        ax.set_title('Homotypic interaction, $M$ (blue), $s_M$=' + str(s_M) +', $X$ (green), $s_X$=' + str(s_X) + ', (s<r = attraction)')
        
        plt.pause(0.01)
    
    ########################## Build ODE RHS ###########################
   
    def f(t, Y):
        
        # Initialise output
        
        dYdt = np.zeros(2*Nx)
        
        rho=0.5
        M=Y[:Nx]
        X=Y[Nx:]
        
        #Probability of movement given that you will make bonds and not break bonds
        p_M=0.25
        p_X=0.25
        
        #Prob of movement given you will not make bonds or break bonds
        s_M=0.25
        s_X=0.25
        
        #Prob of movement given you will make bonds and break bonds
        u_M=0.25
        u_X=0.25
        
        #Prob of movement given that you will break bonds and not make bonds
        q_M=0.25
        q_X=0.25
       
        
        # Manipulate M to get M_j-1 etc. - assuming periodic BCs
        
        M_jplus2 = np.roll(M, -2) # Moving elements back two means we are now indexing the j+2'th element in the j'th place
        
        M_jplus1 = np.roll(M, -1) # Moving elements back one means we are now indexing the j+1'th element in the j'th place
        
        M_jminus1 = np.roll(M, 1) # Moving elements forward one means we are now indexing the j-1'th element in the j'th place
        
        M_jminus2 = np.roll(M, 2) # Moving elements forward two means we are now indexing the j-2'th element in the j'th place
        
        
        X_jplus2 = np.roll(X, -2) # Moving elements back two means we are now indexing the j+2'th element in the j'th place
        
        X_jplus1 = np.roll(X, -1) # Moving elements back one means we are now indexing the j+1'th element in the j'th place
        
        X_jminus1 = np.roll(X, 1) # Moving elements forward one means we are now indexing the j-1'th element in the j'th place
        
        X_jminus2 = np.roll(X, 2) # Moving elements forward two means we are now indexing the j-2'th element in the j'th place
        
        # Calculate RHS
        
        dMdx = (M_jplus2 - M_jminus2)/(4*dx)  # fourth order
        
        d2Mdx2 = (-M_jplus2 + 16*M_jplus1 - 30*M + 16*M_jminus1 - M_jminus2)/(12*dx*dx) # fourth order
        
        dXdx = (X_jplus2 - X_jminus2)/(4*dx)  # fourth order
        
        d2Xdx2 = (-X_jplus2 + 16*X_jplus1 - 30*X + 16*X_jminus1 - X_jminus2)/(12*dx*dx) # fourth order
        
        
        #Heterotypic interaction (s>r = repulsion, s<r = attraction)
        
        #First Nx are M, second Nx are X
        
        #dYdt[:Nx] = (P_m/2.0)*((r_M-s_M)*(X*(1-M-X)*d2Mdx2+M*(X-2*(1-M-X))*d2Xdx2+4*M*dXdx**2+(4*M-2*(1-M-X))*dMdx*dXdx)+s_M*((1-X)*d2Mdx2+M*d2Xdx2))
        
        #dYdt[Nx:] = (P_m/2.0)*((r_X-s_X)*(M*(1-M-X)*d2Xdx2+X*(M-2*(1-M-X))*d2Mdx2+4*X*dMdx**2+(4*X-2*(1-M-X))*dXdx*dMdx)+s_X*((1-M)*d2Xdx2+X*d2Mdx2))         
        
        #Homotypic interaction (s>r = repulsion, s<r = attraction)
        
        #First Nx are M, second Nx are X
        
        #dYdt[:Nx] = (P_m/2.0)*(r_M-s_M)*(M*(2*X+3*M-2)*d2Mdx2+(6*M+2*X-2)*(dMdx)**2+M**2*(d2Xdx2)+4*M*dXdx*dMdx)+(P_m/2.0)*s_M*((1-X)*d2Mdx2+M*d2Xdx2)
        
        #dYdt[Nx:] = (P_m/2.0)*(r_X-s_X)*(X*(2*M+3*X-2)*d2Xdx2+(6*X+2*M-2)*(dXdx)**2+X**2*(d2Mdx2)+4*X*dMdx*dXdx)+(P_m/2.0)*s_X*((1-M)*d2Xdx2+X*d2Mdx2)
        
        #Homotypic interaction with swapping (s>r = repulsion, s<r = attraction)
        
        #dYdt[:Nx] = (P_m/2.0)*(r_M-s_M)*(M*(2*X+3*M-2)*d2Mdx2+(6*M+2*X-2)*(dMdx)**2+M**2*(d2Xdx2)+4*M*dXdx*dMdx)+(P_m/2.0)*s_M*((1-X)*d2Mdx2+M*d2Xdx2)+rho*(P_m/2.0)*((r_M-s_M)*(-2*M*X*d2Mdx2-2*X*dMdx**2-4*M*dXdx*dMdx)+(r_X-s_X)*(4*X*dMdx*dXdx+d2Mdx2*X**2+2*M*(dXdx)**2+2*M*X*d2Xdx2))+rho*(P_m/2.0)*(s_M+s_X)*(X*d2Mdx2-M*d2Xdx2)
        
        #dYdt[Nx:] = (P_m/2.0)*(r_X-s_X)*(X*(2*M+3*X-2)*d2Xdx2+(6*X+2*M-2)*(dXdx)**2+X**2*(d2Mdx2)+4*X*dMdx*dXdx)+(P_m/2.0)*s_X*((1-M)*d2Xdx2+X*d2Mdx2)+rho*(P_m/2.0)*((r_X-s_X)*(-2*M*X*d2Xdx2-2*M*dXdx**2-4*X*dMdx*dXdx)+(r_M-s_M)*(4*M*dXdx*dMdx+d2Xdx2*M**2+2*X*(dMdx)**2+2*M*X*d2Mdx2))+rho*(P_m/2.0)*(s_M+s_X)*(M*d2Xdx2-X*d2Mdx2)
        
        #Heterotypic interaction with swapping (s>r = repulsion, s<r = attraction)
        
        #First Nx are M, second Nx are X
        
        #dYdt[:Nx] = (P_m/2.0)*((r_M-s_M)*(X*(1-M-X)*d2Mdx2+M*(X-2*(1-M-X))*d2Xdx2+4*M*dXdx**2+(4*M-2*(1-M-X))*dMdx*dXdx)+s_M*((1-X)*d2Mdx2+M*d2Xdx2))+rho*(P_m/2.0)*((r_X-s_X)*(M*X*d2Mdx2-2*(dMdx)**2+2*M*dXdx*dMdx-M**2*d2Xdx2)+(r_M-s_M)*(3*M*X*d2Xdx2-4*M*dXdx**2+4*X*dMdx*dXdx+d2Mdx2*X**2)+(s_M+s_X)*(X*d2Mdx2-M*d2Xdx2))
        
        #dYdt[Nx:] = (P_m/2.0)*((r_X-s_X)*(M*(1-M-X)*d2Xdx2+X*(M-2*(1-M-X))*d2Mdx2+4*X*dMdx**2+(4*X-2*(1-M-X))*dXdx*dMdx)+s_X*((1-M)*d2Xdx2+X*d2Mdx2))+rho*(P_m/2.0)*((r_M-s_M)*(M*X*d2Xdx2-2*(dXdx)**2+2*X*dMdx*dXdx-X**2*d2Mdx2)+(r_X-s_X)*(3*M*X*d2Mdx2-4*X*dMdx**2+4*M*dXdx*dMdx+d2Xdx2*M**2)+(s_M+s_X)*(M*d2Xdx2-X*d2Mdx2))        
        
        #Homotypic interaction with swapping (both get option).
        
        #dYdt[:Nx] = (P_m/2.0)*(r_M-s_M)*(M*(2*X+3*M-2)*d2Mdx2+(6*M+2*X-2)*(dMdx)**2+M**2*(d2Xdx2)+4*M*dXdx*dMdx)+(P_m/2.0)*s_M*((1-X)*d2Mdx2+M*d2Xdx2)+rho*P_m*(r_M-s_M)*(r_X-s_X)*(3*X*d2Xdx2*M**2+4*M*d2Mdx2*X**2-2*X**2*(dMdx)**2+2*M*(dXdx)**2-d2Xdx2*M)+rho*P_m*s_X*(r_M-s_M)*(-2*M*X*d2Mdx2-2*(dMdx**2)*X-4*M*dMdx*dXdx+d2Xdx2*M)+rho*P_m*s_M*(r_X-s_X)*(3*M*X*d2Mdx2+4*X*dXdx*dMdx+(X**2)*d2Mdx2-2*(dXdx**2)*M-d2Xdx2*M*X)+rho*P_m*(s_M*s_X)*(X*d2Mdx2-M*d2Xdx2)
        
        #dYdt[Nx:] = (P_m/2.0)*(r_X-s_X)*(X*(2*M+3*X-2)*d2Xdx2+(6*X+2*M-2)*(dXdx)**2+X**2*(d2Mdx2)+4*X*dMdx*dXdx)+(P_m/2.0)*s_X*((1-M)*d2Xdx2+X*d2Mdx2)+rho*P_m*(r_X-s_X)*(r_M-s_M)*(3*X**2*M*d2Mdx2+4*X*M**2*d2Xdx2-2*M**2*(dXdx)**2+2*X*(dMdx)**2-d2Mdx2*X)+rho*P_m*s_M*(r_X-s_X)*(-2*M*X*d2Xdx2-2*(dXdx**2)*M-4*X*dXdx*dMdx+d2Mdx2*X)+rho*P_m*s_X*(r_M-s_M)*(3*M*X*d2Xdx2+4*M*dMdx*dXdx+(M**2)*d2Xdx2-2*(dMdx**2)*X-d2Mdx2*M*X)+rho*P_m*(s_M*s_X)*(M*d2Xdx2-X*d2Mdx2)
        
        #Homotypic interaction without swapping (s_M, u_M)
        
        #dYdt[:Nx]= (P_m/2.0)*((s_M*(1-X)-2*M*(p_M+2*q_M-s_M+X*(p_M-2*q_M+s_M)+M*(2*p_M-4*q_M+s_M+u_M+X*(p_M+q_M-s_M-u_M))))*d2Mdx2-2*(p_M-q_M)*(1-2*M-X)*((dMdx)**2)*M*(s_M+M*(p_M+q_M-2*s_M+M*(s_M+u_M-p_M-q_M))*d2Xdx2+2*M*(p_M-q_M)*dMdx*dXdx))
        
        #dYdt[Nx:]= (P_m/2.0)*((s_X*(1-M)-2*X*(p_X+2*q_X-s_X+M*(p_X-2*q_X+s_X)+X*(2*p_X-4*q_X+s_X+u_X+M*(p_X+q_X-s_X-u_X))))*d2Xdx2-2*(p_X-q_X)*(1-2*X-M)*((dXdx)**2)*X*(s_X+X*(p_X+q_X-2*s_X+X*(s_X+u_X-p_X-q_X))*d2Mdx2+2*X*(p_X-q_X)*dMdx*dXdx))
        
        dYdt[:Nx]=-P_m*(d2Mdx2*(M + X - 1)*(M**2*u_M + s_M*(M - 1)**2 - M*p_M*(M - 1) - M*q_M*(M - 1)) + 2*dMdx*(M + X - 1)*(M*dMdx*q_M - 2*M*dMdx*p_M + M*dMdx*u_M + dMdx*p_M*(M - 1) - 2*dMdx*q_M*(M - 1) + dMdx*s_M*(M - 1)) - 2*M*(dMdx + dXdx)*(M*dMdx*p_M - 2*M*dMdx*q_M + M*dMdx*u_M - 2*dMdx*p_M*(M - 1) + dMdx*q_M*(M - 1) + dMdx*s_M*(M - 1)) - 2*M*(M + X - 1)*(2*dMdx**2*s_M - 2*dMdx**2*q_M - 2*dMdx**2*p_M + 2*dMdx**2*u_M + 2*M*d2Mdx2*p_M + (M*d2Mdx2*q_M)/2 - (5*M*d2Mdx2*u_M)/2 + (d2Mdx2*p_M*(M - 1))/2 + 2*d2Mdx2*q_M*(M - 1) - (5*d2Mdx2*s_M*(M - 1))/2) + 2*M*(M + X - 1)*(2*dMdx**2*s_M - 2*dMdx**2*q_M - 2*dMdx**2*p_M + 2*dMdx**2*u_M + (M*d2Mdx2*p_M)/2 + 2*M*d2Mdx2*q_M - (5*M*d2Mdx2*u_M)/2 + 2*d2Mdx2*p_M*(M - 1) + (d2Mdx2*q_M*(M - 1))/2 - (5*d2Mdx2*s_M*(M - 1))/2) - 2*M*(d2Mdx2/2 + d2Xdx2/2)*(M**2*u_M + s_M*(M - 1)**2 - M*p_M*(M - 1) - M*q_M*(M - 1)))
        
        dYdt[Nx:]=P_m*(2*X*(d2Mdx2/2 + d2Xdx2/2)*(X**2*u_X + s_X*(X - 1)**2 - X*p_X*(X - 1) - X*q_X*(X - 1)) - d2Xdx2*(M + X - 1)*(X**2*u_X + s_X*(X - 1)**2 - X*p_X*(X - 1) - X*q_X*(X - 1)) - 2*dXdx*(M + X - 1)*(X*dXdx*q_X - 2*X*dXdx*p_X + X*dXdx*u_X + dXdx*p_X*(X - 1) - 2*dXdx*q_X*(X - 1) + dXdx*s_X*(X - 1)) + 2*X*(dMdx + dXdx)*(X*dXdx*p_X - 2*X*dXdx*q_X + X*dXdx*u_X - 2*dXdx*p_X*(X - 1) + dXdx*q_X*(X - 1) + dXdx*s_X*(X - 1)) + 2*X*(M + X - 1)*(2*dXdx**2*s_X - 2*dXdx**2*q_X - 2*dXdx**2*p_X + 2*dXdx**2*u_X + 2*X*d2Xdx2*p_X + (X*d2Xdx2*q_X)/2 - (5*X*d2Xdx2*u_X)/2 + (d2Xdx2*p_X*(X - 1))/2 + 2*d2Xdx2*q_X*(X - 1) - (5*d2Xdx2*s_X*(X - 1))/2) - 2*X*(M + X - 1)*(2*dXdx**2*s_X - 2*dXdx**2*q_X - 2*dXdx**2*p_X + 2*dXdx**2*u_X + (X*d2Xdx2*p_X)/2 + 2*X*d2Xdx2*q_X - (5*X*d2Xdx2*u_X)/2 + 2*d2Xdx2*p_X*(X - 1) + (d2Xdx2*q_X*(X - 1))/2 - (5*d2Xdx2*s_X*(X - 1))/2))
        
        return dYdt
    
    ############################# Solve ODE ############################
    
    # Set up ODE integrator
	
    solver = LSODA(f, t_0, Y_0, run_length_max)
    
    # Integrate ODE forward
    
    while solver.t < run_length_max:
        
        # Step solver forward
        
        solver.step()
        
        output = 'Run time = %.6f s.' % solver.t
        
        conc_M=(simps(solver.y[:Nx],x))/init_conc_M0
        conc_X=(simps(solver.y[Nx:],x))/init_conc_X0
        
        
        output += '    '+'Mass conservation M = %.6f %%' % (100*conc_M)
        
        output += '    '+'Mass conservation X = %.6f %%' % (100*conc_X)
        
        Printer(output)

        if plotting:
            
            # Plot title
            
            fig.suptitle('Model at ' + str('{:.2f}'.format(solver.t)) + 's.')
            
            # Plot concentrations on 5 different axes
            
            ax.clear()
            ax.plot(x,M_0,'b--')
            ax.plot(x,X_0,'g--')
            M=solver.y[:Nx]
            X=solver.y[Nx:]
            
            ax.plot(x,M,'b')
            ax.plot(x,X,'g')
            ax.set_title('Homotypic interaction, $M$ (blue), $s_M$=' + str(s_M) +', $X$ (green), $s_X$=' + str(s_X) + ', (s<r = attraction)')
            
            plt.pause(0.01)
    
    ############################## Output ##############################
    
    # Interactive output
    
    Y = solver.y
    
    
    return Y
