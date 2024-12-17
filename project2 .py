""" 
    Your college id here:06014121
    Template code for project 2, contains 9 functions:
    simulate1: complete function for part 1
    part1q1a, part1q1b, part1q1c, part1q2: functions to be completed for part 1
    dualfd1,fd2: complete functions for part 2
    part2q1, part2q2: functions to be completed for part 2
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.linalg
#you may use numpy, scipy, and matplotlib as needed


#---------------------------- Code for Part 1 ----------------------------#
def simulate1(n,X0=-1,Nt=2000,tf=1000,g=1.0,mu=0.0,flag_display=False):
    """
    Simulation code for part 1
    Input:
        n: number of ODEs
        X0: initial condition if n-element numpy array. If X0 is an integer, the code will set 
        the initial condition (otherwise an array is expected).
        Nt: number of time steps
        tf: final time
        g,mu: model parameters
        flag_display: will generate a contour plot displaying results if True
    Output:
    t,x: numpy arrays containing times and simulation results
    """

    def model(t, X, n, g, mu):
        """
        Defines the system of ODEs 
        
        Input:
        - t: time 
        - X: array of variables [x_0, x_1, ..., x_{n-1}]
        - g, mu: scalar model parameters
        - n: number of ODEs (size of the system)
         
        Returns:
        - dXdt: array of derivatives [dx_0/dt, dx_1/dt, ..., dx_{n-1}/dt]
        """
        dXdt = np.zeros(n)
        dXdt[0] = X[0]*(1-g*X[-2]-X[0]-g*X[1]+mu*X[-3])
        dXdt[1] = X[1]*(1-g*X[-1]-X[1]-g*X[2])
        dXdt[2:n-1] = X[2:n-1]*(1-g*X[0:n-3]-X[2:n-1]-g*X[3:n])
        dXdt[-1] = X[-1]*(1-g*X[-3]-X[-1]-g*X[0])
        return dXdt

    # Parameters
    t_span = (0, tf)  # Time span for the integration
    if type(X0)==int: #(modified from original version)
        X0 = 1/(2*g+1)+0.001*np.random.rand(n)  # Random initial conditions, modify if needed

    # Solve the system
    solution = solve_ivp(
        fun=model,
        t_span=t_span,
        y0=X0,
        args=(n, g, mu),
        method='BDF',rtol=1e-10,atol=1e-10,
        t_eval=np.linspace(t_span[0], t_span[1], Nt)  # Times to evaluate the solution
    )

    t,x = solution.t,solution.y #(in original version of code this line was inside if-block below)
    if flag_display:
        # Plot the solution
        plt.contour(t,np.arange(n),x,20)
        plt.xlabel('t') #(corrected axis labels)
        plt.ylabel('i')
    return t,x

def part1q1a(n,g,mu,T):
    """Part 1, question 1 (a)
    Use the variable inputs if/as needed.
    Input:
    n: number of ODEs
    g,mu: model parameters
    T: time at which perturbation energy ratio should be maximized
    
    Output:
    xbar: n-element array containing non-trivial equilibrium solution
    xtilde0: n-element array corresponding to computed initial condition
    eratio: computed maximum perturbation energy ratio
    """
    #use/modify code below as needed:
    xbar = np.zeros(n)
    xtilde0 = np.zeros(n)
    eratio = 0.0 #should be modified below
    

    #add code here
    from scipy.optimize import root 
    def model(X, n, g, mu):
        """
        Defines the system of ODEs 
        
        Input:
        - t: time 
        - X: array of variables [x_0, x_1, ..., x_{n-1}]
        - g, mu: scalar model parameters
        - n: number of ODEs (size of the system)
         
        Returns:
        - dXdt: array of derivatives [dx_0/dt, dx_1/dt, ..., dx_{n-1}/dt]
        """
        dXdt = np.zeros(n)
        dXdt[0] = X[0]*(1-g*X[-2]-X[0]-g*X[1]+mu*X[-3])
        dXdt[1] = X[1]*(1-g*X[-1]-X[1]-g*X[2])
        dXdt[2:n-1] = X[2:n-1]*(1-g*X[0:n-3]-X[2:n-1]-g*X[3:n])
        dXdt[-1] = X[-1]*(1-g*X[-3]-X[-1]-g*X[0])
        return dXdt
    
    
        
    X0 = 1/(2*g+1)+0.001*np.random.rand(n)
    solution = root(model, X0,args= (n,g,mu), method='hybr' )#using root to find non trivial equilibruim point
    if solution.success:
        xbar = solution.x
    else:
        raise ValueError("failed to find equilibrium point")
    
    """convert ODEs into matrix form"""
    
    B = -np.eye(n)
    for i in range(n):
        if i == 0:
            
            B[i, -2] = -g 
            B[i, 1] = -g  
            B[i, -3] = mu
        elif i == 1:
            
            B[i, -1] = -g  
            B[i, 2] = -g   
        elif i == n - 1:
            
            B[i, -3] = -g 
            B[i, 0] = -g   
        else:
            
            B[i, i - 2] = -g
            B[i, (i + 1)] = -g  
    
   
    
    M=  np.diag(xbar)@B
    A = scipy.linalg.expm(M*T)
    
    U,S,VT = np.linalg.svd(A)
    eratio= S.max()**2
    xtilde0 = VT[0,:]
    
    
    return xbar,xtilde0,eratio



    

def part1q1b():
    """Part 1, question 1(b): 
    Add input/output if/as needed.
    """
    #use/modify code below as needed:
    
    #add code here
    def test1():#accuracy for maximum purturbation       
        eratio_list = []
        eratiosim_list = []
        error_list = []
        for i in np.arange(1,1000):    
            n = 19
            g = 1.2
            mu = 2.5
            T = 50
            tf=T
            Nt=1000
            
            xbar,xtilde0,eratio = part1q1a(n,g,mu,T)
            e = 1e-6
            X0 = xbar + e*(xtilde0/np.linalg.norm(xtilde0))
            t,x = simulate1(n,X0,Nt,tf,g,mu,flag_display=False)
            xtildeT = x[:,-1] - xbar
            eratiosim = (np.linalg.norm(xtildeT)/np.linalg.norm(e*xtilde0))**2
            
            error = abs(eratio - eratiosim)/eratio
            eratio_list.append(eratio)
            eratiosim_list.append(eratiosim)
            error_list.append(error)
        
        
    
        
    
        avg_eratio = np.mean(eratio_list)
        avg_eratiosim = np.mean(eratiosim_list)
        avg_error = np.mean(error_list)
        
        return avg_eratio, avg_eratiosim, avg_error
    def test2():#accuracy or multiple directions
        
        n = 19
        g = 1.2
        mu = 2.5
        T = 50
        tf=T
        Nt=1000
        e = 1e-6
        eratio_list = []
        eratiosim_list = []
        error_list = []
        def part1q1a(n,g,mu,T):
          
            
            xbar = np.zeros(n)
            xtilde0 = np.zeros(n)
            
            
            from scipy.optimize import root 
            def model(X, n, g, mu):
                
                dXdt = np.zeros(n)
                dXdt[0] = X[0]*(1-g*X[-2]-X[0]-g*X[1]+mu*X[-3])
                dXdt[1] = X[1]*(1-g*X[-1]-X[1]-g*X[2])
                dXdt[2:n-1] = X[2:n-1]*(1-g*X[0:n-3]-X[2:n-1]-g*X[3:n])
                dXdt[-1] = X[-1]*(1-g*X[-3]-X[-1]-g*X[0])
                return dXdt
            
            
                
            X0 = 1/(2*g+1)+0.001*np.random.rand(n)
            solution = root(model, X0,args= (n,g,mu), method='hybr' )
            if solution.success:
                xbar = solution.x
            else:
                raise ValueError("failed to find equilibrium point")
            
            
            
            B = -np.eye(n)
            for i in range(n):
                if i == 0:
                    
                    B[i, -2] = -g 
                    B[i, 1] = -g  
                    B[i, -3] = mu
                elif i == 1:
                    
                    B[i, -1] = -g  
                    B[i, 2] = -g   
                elif i == n - 1:
                    
                    B[i, -3] = -g 
                    B[i, 0] = -g   
                else:
                    
                    B[i, i - 2] = -g
                    B[i, (i + 1)] = -g  
            
           
            
            M=  np.diag(xbar)@B
            A = scipy.linalg.expm(M*T)
            
            U,S,VT = np.linalg.svd(A)
           
            
            
            return xbar,VT,S
        xbar,VT,S = part1q1a(n,g,mu,T)
                
        for i in range(n):
            xtilde0 = VT[i, :]
            eratio = S[i] ** 2

            # Apply perturbation along this singular vector
            perturbation = e * xtilde0 / np.linalg.norm(xtilde0)
            X0 = xbar + perturbation

            t, x = simulate1(n, X0, Nt, tf, g, mu, flag_display=False)
            xtildeT = x[:, -1] - xbar
            eratiosim = (np.linalg.norm(xtildeT) / np.linalg.norm(perturbation)) ** 2

            error = np.abs(eratio - eratiosim)
            eratio_list.append(eratio)
            eratiosim_list.append(eratiosim)
            error_list.append(error)

        
        

        

        avg_eratio = np.mean(eratio_list)
        avg_eratiosim = np.mean(eratiosim_list)
        avg_error = np.mean(error_list)
        
        return avg_eratio, avg_eratiosim, avg_error

    return test1(),test2() #modify if needed


def part1q1c():
    """Part 1, question 1(c): 
    Add input/output if/as needed.
    """
    #use/modify code below as needed:
    n = 19
    g = 2
    mu = 0

    #add code here
    n = 19
    g = 2
    mu = 0
    Ts=[]
    eratios=[]
    eratiossim = []
    errors = []
    Nt = 1000
    e = 1e-6
    for T in np.linspace(1, 50,1000):
        xbar,xtilde0,eratio = part1q1a(n,g,mu,T)
        
        
        tf = T
       
        X0 = xbar + e*(xtilde0/np.linalg.norm(xtilde0))
        t,x = simulate1(n,X0,Nt,tf,g,mu,flag_display=False)
      
        xtildeT = x[:,-1] - xbar
        
        eratiosim = (np.linalg.norm(xtildeT)/np.linalg.norm(e*xtilde0))**2
        error  = abs(eratio - eratiosim)/eratio
        Ts.append(T)
        eratios.append(eratio)
        eratiossim.append(eratiosim)
        errors.append(error)
    
    plt.figure()
    plt.plot(Ts,eratios)
    plt.title('eratio vs T')
    plt.xlabel('T')
    plt.ylabel('eratio')
    plt.show()
    
    plt.figure()
    a,b = np.polyfit(Ts,np.log(eratios),1)
    print(b)
    plt.semilogy(Ts, eratios,label=' a =%f'%a)
    plt.title('eratio vs T')
    plt.xlabel('T')
    plt.ylabel('log(eratio)')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(Ts,eratiossim)
    plt.title('eratiosim vs T')
    plt.xlabel('T')
    plt.ylabel('eratiosim')
    plt.show()
    
    plt.figure()
    a,b = np.polyfit(Ts,np.log(eratiossim),1)
    print(b)
    plt.semilogy(Ts, eratiossim,label=' a =%f'%a)
    plt.title('eratiosim vs T')
    plt.xlabel('T')
    plt.ylabel('log(eratiosim)')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(Ts,errors)
    plt.xlabel('T')
    plt.ylabel('error')
    plt.title('error plot')
    plt.show()
    
    #check max eigenvalue 
    xbar,xtilde0,eratio =part1q1a(n,g,mu,T=1)
    print(eratio)


    return None #modify if needed

def part1q2():
    """Part 1, question 2: 
    Add input/output if/as needed.
    """
    from scipy.signal import welch
    from scipy.spatial.distance import pdist 

    #add code here
    # plot all contours to find where systems are periodic
    plt.figure(dpi=250)
    simulate1(9,X0=-1,Nt=2000,tf=200,g=1.0,mu=0.0,flag_display=True)
    plt.title('n=9')
    plt.show()
    plt.figure(dpi=250)
    simulate1(20,X0=-1,Nt=2000,tf=3000,g=1.0,mu=0.0,flag_display=True)
    plt.title('n=20')
    plt.show()
    plt.figure(dpi=250)
    simulate1(59,X0=-1,Nt=2000,tf=3000,g=1.0,mu=0.0,flag_display=True)
    plt.title("n=59")
    plt.show()
    
    #correlation heatmaps
    nskip = 100
    ta,xa = simulate1(9,X0=-1,Nt=200,tf=200,g=1.0,mu=0.0,flag_display=False)
    xa = xa[:,nskip:]
    ta = ta[nskip:]
    cor_matrix =np.corrcoef(xa)
    
    
    plt.figure()
    
    plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest',origin='lower')
    plt.colorbar(label='Correlation Coefficient')

    plt.title('n=9')
    plt.show()
    
    nskip = 400
    tb,xb = simulate1(20,X0=-1,Nt=1000,tf=1000,g=1.0,mu=0.0,flag_display=False)
    xb = xb[:,nskip:]
    tb = tb[nskip:]
    cor_matrix =np.corrcoef(xb)
    plt.figure()
    
    plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest',origin='lower')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('n=20,tf<1000')

    
    plt.show()
    
    nskip = 1500
    tb,xb = simulate1(20,X0=-1,Nt=3000,tf=3000,g=1.0,mu=0.0,flag_display=False)
    xb = xb[:,nskip:]
    tb = tb[nskip:]
    cor_matrix =np.corrcoef(xb)
    plt.figure()
    
    plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest',origin='lower')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('n=20, tf>1500')
    plt.show()
    
    nskip = 500
    tc,xc = simulate1(59,X0=-1,Nt=3000,tf=2000,g=1.0,mu=0.0,flag_display=False)
    xc = xc[:,nskip:]
    tc = tc[nskip:]
    cor_matrix =np.corrcoef(xc)
    plt.figure()
    
    plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest',origin='lower')
    plt.colorbar(label='Correlation Coefficient') 
    plt.title('n=59')
    plt.show()
    
    #sinusodal plots
    
    nskip = 50
    ta,xa = simulate1(9,X0=-1,Nt=100,tf=100,g=1.0,mu=0.0,flag_display=False)
    
    xa = xa.T[nskip:,:]
    ta = ta[nskip:]
    plt.plot(ta,xa)
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=9")
    plt.show()
    
    
    
   
    nskip = 900
    tb,xb = simulate1(20,X0=-1,Nt=1000,tf=1000,g=1.0,mu=0.0,flag_display=False)
    
    xb = xb.T[nskip:,:]
    tb = tb[nskip:]
    plt.plot(tb,xb)
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=20")
    
    plt.show()
    
    nskip = 1900
    tb,xb = simulate1(20,X0=-1,Nt=2000,tf=2000,g=1.0,mu=0.0,flag_display=False)
    
    xb = xb.T[nskip:,:]
    tb = tb[nskip:]
    plt.plot(tb,xb)
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=20")
    plt.show()
    
   
    nskip = 1900
    tc,xc = simulate1(59,X0=-1,Nt=2000,tf=2000,g=1.0,mu=0.0,flag_display=False)
    
    xc = xc.T[nskip:,:]
    tc = tc[nskip:]
    plt.plot(tc,xc)
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=59")
    plt.show()
    plt.plot(tc,xc[:,-1])
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=59")
    plt.show()
    #phase diagrams
    nskip = 500
    ta,xa = simulate1(9,X0=-1,Nt=2000,tf=1000,g=1.0,mu=0.0,flag_display=False)
    
    xa = xa[:,nskip:]
    ta = ta[nskip:]
    
    plt.figure()
    plt.plot(xa[0],xa[1])
    plt.title('n=9')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(xa[0],xa[1],xa[2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('n=9')
    plt.show()
   
    nskip = 500
    tb,xb = simulate1(20,X0=-1,Nt=2000,tf=1000,g=1.0,mu=0.0,flag_display=False)
    
    xb = xb[:,nskip:]
    tb = tb[nskip:]
    plt.figure()
    plt.plot(xb[0],xb[2])
    plt.title('n=20, tf<1000')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(xb[0],xb[1],xb[2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('n=20, tf<1000')
    plt.show()
    
    nskip = 1500
    tb,xb = simulate1(20,X0=-1,Nt=2000,tf=2000,g=1.0,mu=0.0,flag_display=False)
    
    xb = xb[:,nskip:]
    tb = tb[nskip:]
    plt.figure()
    plt.plot(xb[0],xb[2])
    plt.title('n=20, tf>1500')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(xb[0],xb[1],xb[2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('n=20, tf>1500')
    plt.show()
    
   
    nskip = 500
    tc,xc = simulate1(59,X0=-1,Nt=2000,tf=2000,g=1.0,mu=0.0,flag_display=False)
    
    xc = xc[:,nskip:]
    tc = tc[nskip:]
    plt.figure()
    plt.plot(xc[0],xc[2])
    plt.title('n=59')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(xc[0],xc[1],xc[2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('n=59')
    plt.show()
    
    """ case A """
    #np.random.seed(16012002)
    nskip = 100
    ta,xa = simulate1(9,X0=-1,Nt=2000,tf=1000,g=1.0,mu=0.0,flag_display=False)
    xa = xa.T[nskip:,-1]
    ta = ta[nskip:]
    plt.plot(ta,xa)
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=9")
    
    plt.show()
    #welch to find dominant frequency, tells us about periodicity
    Xxx,Pxx = welch(xa,fs=1/(ta[1]-ta[0]))
    plt.figure()
    plt.semilogy(Xxx,Pxx)
    plt.title('Power spectrum of x, n=9')
    plt.xlabel(r'$f$')
    plt.ylabel(r'$P_{xx}$')
    plt.grid()
    plt.show()
    f = Xxx[Pxx==Pxx.max()][0]
    print("fa=",f)
    print("dta,1/fa=",ta[1]-ta[0],1/f)
    tau = 1/(5*f)
    
    Del = int(tau/(ta[1]-ta[0]))
    v1 = np.vstack([xa[:-2*Del],xa[Del:-Del],xa[2*Del:]])
    A = v1.T
    D = pdist(A)
    eps = np.logspace(-6, -2,200)
    C = np.zeros_like(eps)
    

    for i,j in enumerate(eps):    
        E = D[D<j]
        C[i]= E.size/len(D)
    
    l = 10**(-5)
    u= 10**(-3.5)


    mask = (eps >= l) & (eps <= u)
    plt.figure()
    plt.loglog(eps,C,'x',label = r'$C(\epsilon)$')
    a,b = np.polyfit(np.log(eps[mask]),np.log(C[mask]),1)
    plt.loglog(eps,np.exp(b)*eps**a,'k:',label='Least squares fit, a =%f'%a)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$C(\epsilon)$')
    plt.title('n=9')
    plt.legend()
    plt.show()
    
    
    
    """ case B when tf<1000"""
    
    nskip = 400
    
    tb,xb = simulate1(20,X0=-1,Nt=1000,tf=1000,g=1.0,mu=0.0,flag_display=False)
    
    xb = xb.T[nskip:,-1]
    tb = tb[nskip:]
    plt.plot(tb,xb)
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=20")
    plt.show()
    #welch to find dominant frequency, tells us about periodicity
    Xxx,Pxx = welch(xb,fs=1/(tb[1]-tb[0]))
    plt.figure()
    plt.semilogy(Xxx,Pxx)
    plt.title('Power spectrum of x, n=20, tf<1000')
    plt.xlabel(r'$f$')
    plt.ylabel(r'$P_{xx}$')
    plt.grid()
    plt.show()
    f = Xxx[Pxx==Pxx.max()][0]
    print("fb1=",f)
    print("dtb1,1/fb1=",tb[1]-tb[0],1/f)
    tau = 1/(5*f)
    
    Del = int(tau/(tb[1]-tb[0]))
    v1 = np.vstack([xb[:-2*Del],xb[Del:-Del],xb[2*Del:]])
    A = v1.T
    D = pdist(A)
    eps = np.logspace(-5, 1,200)
    C = np.zeros_like(eps)
    

    for i,j in enumerate(eps):    
        E = D[D<j]
        C[i]= E.size/len(D)
    
    l = 10**(-3)
    u= 10**(0)


    mask = (eps >= l) & (eps <= u)
    plt.figure()
    plt.loglog(eps,C,'x',label = r'$C(\epsilon)$')
    a,b = np.polyfit(np.log(eps[mask]),np.log(C[mask]),1)
    plt.loglog(eps,np.exp(b)*eps**a,'k:',label='Least squares fit, a =%f'%a)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$C(\epsilon)$')
    plt.title('n=20,tf<1000')
    plt.legend()
    plt.show()
    
    
    """ case B when 1500<tf<3000"""
    nskip = 1500
    
    tb,xb = simulate1(20,X0=-1,Nt=3000,tf=3000,g=1.0,mu=0.0,flag_display=False)
    
    xb = xb.T[nskip:,-1]
    tb = tb[nskip:]
    plt.plot(tb,xb)
    plt.xlabel('t' )
    plt.ylabel('x')
    plt.title("n=20")
    plt.show()
    #welch to find dominant frequency, tells us about periodicity
    Xxx,Pxx = welch(xb,fs=1/(tb[1]-tb[0]))
    plt.figure()
    plt.semilogy(Xxx,Pxx)
    plt.title('Power spectrum of x, n=20, tf>1500')
    plt.xlabel(r'$f$')
    plt.ylabel(r'$P_{xx}$')
    
    plt.grid()
    plt.show()
    f = Xxx[Pxx==Pxx.max()][0]
    print("fb2=",f)
    print("dtb2,1/fb2=",tb[1]-tb[0],1/f)
    tau = 1/(5*f)
    
    Del = int(tau/(tb[1]-tb[0]))
    v1 = np.vstack([xb[:-2*Del],xb[Del:-Del],xb[2*Del:]])
    A = v1.T
    D = pdist(A)
    eps = np.logspace(-5, 2,200)
    C = np.zeros_like(eps)
    

    for i,j in enumerate(eps):    
        E = D[D<j]
        C[i]= E.size/len(D)
    
    l = 10**(-3)
    u= 10**(0)


    mask = (eps >= l) & (eps <= u)
    plt.figure()
    plt.loglog(eps,C,'x',label = r'$C(\epsilon)$')
    a,b = np.polyfit(np.log(eps[mask]),np.log(C[mask]),1)
    plt.loglog(eps,np.exp(b)*eps**a,'k:',label='Least squares fit, a =%f'%a)\
    
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$C(\epsilon)$')
    plt.title('n=20,tf>1500')
    plt.legend()
    plt.show()
    
    """case C"""
    nskip=500
    tc,xc = simulate1(59,X0=-1,Nt=3000,tf=3000,g=1.0,mu=0.0,flag_display=False)
    
    xc = xc.T[nskip:,-1]
    tc = tc[nskip:]
    plt.plot(tc,xc)
    plt.title('n=59')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()
    #welch to find dominant frequency, tells us about periodicity
    Xxx,Pxx = welch(xc,fs=1/(tc[1]-tc[0]))
    plt.figure()
    plt.semilogy(Xxx,Pxx)
    plt.xlabel(r'$f$')
    plt.ylabel(r'$P_{xx}$')
    plt.grid()
    plt.title('Power spectrum of x, n=59')
    plt.show()
    f = Xxx[Pxx==Pxx.max()][0]
    print("fc=",f)
    print("dtc,1/fc=",tc[1]-tc[0],1/f)
    
    tau = 1/(5*f)
    
    Del = int(tau/(tc[1]-tc[0]))
    v1 = np.vstack([xc[:-2*Del],xc[Del:-Del],xc[2*Del:]])
    A = v1.T
    D = pdist(A)
    eps = np.logspace(-3, 1,200)
    C = np.zeros_like(eps)
    

    for i,j in enumerate(eps):    
        E = D[D<j]
        C[i]= E.size/len(D)
    
    l = 10**(-2)
    u= 10**(-0.7)


    mask = (eps >= l) & (eps <= u)
    plt.figure()
    plt.loglog(eps,C,'x',label = r'$C(\epsilon)$')
    a,b = np.polyfit(np.log(eps[mask]),np.log(C[mask]),1)
    plt.loglog(eps,np.exp(b)*eps**a,'k:',label='Least squares fit, a =%f'%a)\
    
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$C(\epsilon)$')
    plt.title('n=59')
    plt.legend()
    plt.show()
    

    return None #modify if needed



#---------------------------- End code for Part 1 ----------------------------#


#---------------------------- Code for Part 2 ----------------------------#
def dualfd1(f):
    """
    Code implementing implicit finite difference scheme for special case m=1
    Implementation is not efficient.
    Input:
        f: n-element numpy array
    Output:
        df, d2f: computed 1st and 2nd derivatives
    """
    #parameters, grid
    n = f.size
    h = 1/(n-1)
    x = np.linspace(0,1,n)
    
    #fd method coefficients
    #interior points:
    L1 = [7,h,16,0,7,-h]
    L2 = [-9,-h,0,8*h,9,-h]
    
    #boundary points:
    L1b = [1,0,2,-h]
    L2b = [0,h,-6,5*h]

    L1b2 = [2,h,1,0]
    L2b2 = [-6,-5*h,0,-h]

    A = np.zeros((2*n,2*n))
    #iterate filling a row of A each iteration
    for i in range(n):
        #rows 0 and N-1
        if i==0:
            #Set boundary eqn 1
            A[0,0:4] = L1b
            #Set boundary eqn 2
            A[1,0:4] = L2b
        elif i==n-1:
            A[-2,-4:] = L1b2
            A[-1,-4:] = L2b2
        else:
            #interior rows
            #set equation 1
            ind = 2*i
            A[ind,ind-2:ind+4] = L1
            #set equation 2
            A[ind+1,ind-2:ind+4] = L2

    #set up RHS
    b = np.zeros(2*n)
    c31,c22,cb11,cb21,cb31,cb12,cb22,cb32 = 15/h,24/h,-3.5/h,4/h,-0.5/h,9/h,-12/h,3/h
    for i in range(n):
        if i==0:
            b[i] = cb11*f[0]+cb21*f[1]+cb31*f[2]
            b[i+1] = cb12*f[0]+cb22*f[1]+cb32*f[2]
        elif i==n-1:
            b[-2] =-(cb11*f[-1]+cb21*f[-2]+cb31*f[-3])
            b[-1] = -(cb12*f[-1]+cb22*f[-2]+cb32*f[-3])
        else:
            ind = 2*i
            b[ind] = c31*(f[i+1]-f[i-1])
            b[ind+1] = c22*(f[i-1]-2*f[i]+f[i+1])
    out = np.linalg.solve(A,b)
    df = out[::2]
    d2f = out[1::2]
    return df,d2f


def fd2(f):
    """
    Computes the first and second derivatives with respect to x using second-order finite difference methods.
    
    Input:
    f: m x n array whose 1st and 2nd derivatives will be computed with respect to x
    
    Output:
     df, d2f: m x n arrays conaining 1st and 2nd derivatives of f with respect to x
    """

    m,n = f.shape
    h = 1/(n-1)
    df = np.zeros_like(f) 
    d2f = np.zeros_like(f)
    
    # First derivative 
    # Centered differences for the interior 
    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)

    # One-sided differences at the boundaries
    df[:, 0] = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * h)
    df[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * h)
    
    # Second derivative 
    # Centered differences for the interior 
    d2f[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (h**2)
    
    # One-sided differences at the boundaries
    d2f[:, 0] = (2 * f[:, 0] - 5 * f[:, 1] + 4 * f[:, 2] - f[:, 3]) / (h**2)
    d2f[:, -1] = (2 * f[:, -1] - 5 * f[:, -2] + 4 * f[:, -3] - f[:, -4]) / (h**2)
    
    return df, d2f






def part2q1(f):
    """
    Part 2, question 1
    Input:
        f:  nxm array whose 1st and 2nd derivatives will be computed with respect to x
    Output:
        df, d2f: nxm arrays conaining 1st and 2nd derivatives of f with respect to x
        computed with implicit fd scheme
    """
    #use code below if/as needed
    #generate grid
    m,n = f.shape
    h = 1/(n-1)
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)  
    
   

    
    
    #construction of A in banded form B
    
    B = np.zeros((7,2*n))

    B[0,3::2] = -h
    B[1,2] = 2 
    B[1,3] = 5*h
    B[1,4::2] = 7
    B[1,5::2] = -h 
    B[2,2] = -6
    B[2,4:-1:2] = 9
    B[3,0] = 1 
    B[3,1] = h
    B[3,2:-3:2] = 16
    B[3,3:-2:2] = 8*h
    B[3,-2]= 1
    B[3,-1] = -h
    B[4,1:-1:2] = h
    B[5,0:-5:2]=7
    B[5,1:-4:2] = -h
    B[5,-4] = 2
    B[5,-3]= -5*h
    B[6,0:-5:2] = -9
    B[6,-5] = 0
    B[6,-4] = -6

    

    
    #construction of b ( RHS of equation)

    
    b=np.zeros((2*n,m))
    c31,c22,cb11,cb21,cb31,cb12,cb22,cb32 = 15/h,24/h,-3.5/h,4/h,-0.5/h,9/h,-12/h,3/h

    b[0,:] = cb11*f[:,0]+cb21*f[:,1]+cb31*f[:,2]
    b[1,:] = cb12*f[:,0]+cb22*f[:,1]+cb32*f[:,2]

    b[-2,:] =-(cb11*f[:,-1]+cb21*f[:,-2]+cb31*f[:,-3])
    b[-1,:] = -(cb12*f[:,-1]+cb22*f[:,-2]+cb32*f[:,-3])
    i = np.arange(1,n-1 )
    ind = 2*i
    b[ind,:] = (c31*(f[:,i+1]-f[:,i-1])).T
    b[ind+1,:] = (c22*(f[:,i-1]-2*f[:,i]+f[:,i+1])).T
    
   
    
    out = scipy.linalg.solve_banded((3, 3), B, b)
    df = out[::2, :].T
    d2f = out[1::2, :].T
    
   
    


    




    return df,d2f 
import time 

def part2q2():
    """
    Part 2, question 2
    Add input/output as needed"""

   
    
    def test1():#how n effects the cost and accuracy
        errors_df_fd2 = []
        errors_d2f_fd2 = []
        errors_df_part2q1 = []
        errors_d2f_part2q1 = []
        times_fd2 = []
        times_part2q1 = []
        ns = []
        hs = []
        m =1000
        erors1 = []
        
       
        for n in np.arange(10,2000,100):
            x = np.linspace(0,1,n)
            y = np.linspace(0,1,m)  
            X, Y = np.meshgrid(x, y)
            h = 1/(n-1)
            hs.append(h)
            ns.append(n)
            
            f = np.sin(np.pi * X) * np.cos(np.pi * Y)
            df_exact = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)   # First derivative with respect to x
            d2f_exact = - (np.pi ** 2) * np.sin(np.pi * X) * np.cos(np.pi * Y)
            
            
            t1 = time.time()
            df_p1,d2f_p1 = part2q1(f)
            dt1 = time.time() - t1
            times_part2q1.append(dt1)
            
            t2 = time.time()
            df_fd2,d2f_fd2 = fd2(f)
            dt2 = time.time() - t2
            times_fd2.append(dt2)
            
            
            errordf_fd2 = np.max(np.abs((df_fd2 - df_exact)))
            errors_df_fd2.append(errordf_fd2)
            errord2f_fd2 = np.max(np.abs((d2f_fd2 - d2f_exact)))
            errors_d2f_fd2.append(errord2f_fd2)
            
            errordf_p1 = np.max(np.abs((df_p1 - df_exact)))
            errors_df_part2q1.append(errordf_p1)
            errord2f_p1 = np.max(np.abs((d2f_p1 - d2f_exact)))
            errors_d2f_part2q1.append(errord2f_p1)
        
      
        a,b = np.polyfit(np.log(hs),np.log(errors_df_fd2),1)
        plt.loglog(hs, errors_df_fd2, '-', label='fd2 df error, a =%f'%a)
        plt.loglog(hs,np.exp(b)*hs**a,'k:')
        
        a,b = np.polyfit(np.log(hs),np.log(errors_d2f_fd2),1)
        plt.loglog(hs, errors_d2f_fd2, '-', label='fd2 d2f error, a =%f'%a)
        plt.loglog(hs,np.exp(b)*hs**a,'k:')
        
        a,b = np.polyfit(np.log(hs),np.log(errors_df_part2q1),1)
        plt.loglog(hs, errors_df_part2q1, '-', label='part2q1 df error, a =%f'%a)
        plt.loglog(hs,np.exp(b)*hs**a,'k:')
        
        a,b = np.polyfit(np.log(hs),np.log(errors_d2f_part2q1),1)
        plt.loglog(hs, errors_d2f_part2q1, '-', label='part2q1 d2f error, a =%f'%a)
        plt.loglog(hs,np.exp(b)*hs**a,'k:')
        
        plt.xlabel('n')
        plt.ylabel('Error ')
        plt.legend()
        plt.title('Error vs h')
        plt.grid()
        
        
        plt.figure()
        
        a,b = np.polyfit(np.log(ns),np.log(times_fd2),1)
        plt.loglog(ns, times_fd2, '-', label='fd2 time, a=%f'%a)
        plt.loglog(ns,np.exp(b)*ns**a,'k:')
        
        a,b = np.polyfit(np.log(ns),np.log(times_part2q1),1)
        plt.loglog(ns, times_part2q1, '-', label='part2q1 time, a =%f'%a)
        plt.loglog(ns,np.exp(b)*ns**a,'k:')
        
        plt.xlabel('n (number of x points)')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.title('Computation Time vs n')
        plt.grid()
        
        plt.figure()
        
        a,b = np.polyfit(np.log(ns),np.log(errors_df_fd2),1)
        plt.loglog(ns, errors_df_fd2, '-', label='fd2 df error, a =%f'%a)
        plt.loglog(ns,np.exp(b)*ns**a,'k:')
        
        a,b = np.polyfit(np.log(ns),np.log(errors_d2f_fd2),1)
        plt.loglog(ns, errors_d2f_fd2, '-', label='fd2 d2f error, a =%f'%a)
        plt.loglog(ns,np.exp(b)*ns**a,'k:')
        
        a,b = np.polyfit(np.log(ns),np.log(errors_df_part2q1),1)
        plt.loglog(ns, errors_df_part2q1, '-', label='part2q1 df error, a =%f'%a)
        plt.loglog(ns,np.exp(b)*ns**a,'k:')
        
        a,b = np.polyfit(np.log(ns),np.log(errors_d2f_part2q1),1)
        plt.loglog(ns, errors_d2f_part2q1, '-', label='part2q1 d2f error, a =%f'%a)
        plt.loglog(ns,np.exp(b)*ns**a,'k:')
        
        plt.xlabel('n')
        plt.ylabel('Error ')
        plt.legend()
        plt.title('Error vs n')
        plt.grid()
        
        
           
        return None
    
    def test2():#how m effects the cost and accuracy 
        
        errors_df_fd2 = []
        errors_d2f_fd2 = []
        errors_df_part2q1 = []
        errors_d2f_part2q1 = []
        times_fd2 = []
        times_part2q1 = []
        ms = []
        hs = []
        n =1000
        h = 1/(n-1)
        for m in np.arange(10,3000,100):
            x = np.linspace(0,1,n)
            y = np.linspace(0,1,m)  
            X, Y = np.meshgrid(x, y)
            
            hs.append(h)
            ms.append(m)
            
            f = np.sin(np.pi * X) * np.cos(np.pi * Y)
            df_exact = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)   # First derivative with respect to x
            d2f_exact = - (np.pi ** 2) * np.sin(np.pi * X) * np.cos(np.pi * Y)
            
            t1 = time.time()
            df_p1,d2f_p1 = part2q1(f)
            dt1 = time.time() - t1
            times_part2q1.append(dt1)
            
            t2 = time.time()
            df_fd2,d2f_fd2 = fd2(f)
            dt2 = time.time() - t2
            times_fd2.append(dt2)
            
            errordf_fd2 = np.max(np.abs((df_fd2 - df_exact)))
            errors_df_fd2.append(errordf_fd2)
            errord2f_fd2 = np.max(np.abs((d2f_fd2 - d2f_exact)))
            errors_d2f_fd2.append(errord2f_fd2)
            
            errordf_p1 = np.max(np.abs((df_p1 - df_exact)))
            errors_df_part2q1.append(errordf_p1)
            errord2f_p1 = np.max(np.abs((d2f_p1 - d2f_exact)))
            errors_d2f_part2q1.append(errord2f_p1)
            
            
        
        
        plt.figure()
        
        a,b = np.polyfit(np.log(ms),np.log(times_fd2),1)
        plt.loglog(ms, times_fd2, '-', label='fd2 time, a=%f'%a)
        plt.loglog(ms,np.exp(b)*ms**a,'k:')
        
        a,b = np.polyfit(np.log(ms),np.log(times_part2q1),1)
        plt.plot(ms, times_part2q1, '-', label='part2q1 time, a =%f'%a)
        plt.loglog(ms,np.exp(b)*ms**a,'k:')
        
        
        
        plt.xlabel('m ')
        plt.ylabel('Time')
        plt.legend()
        plt.title('Computation Time vs m')
        plt.grid()
        
        plt.figure()
        
        a,b = np.polyfit(np.log(ms),np.log(errors_df_fd2),1)
        plt.loglog(ms, errors_df_fd2, '-', label='fd2 df error, a =%f'%a)
        plt.loglog(ms,np.exp(b)*ms**a,'k:')
        
        
        a,b = np.polyfit(np.log(ms),np.log(errors_d2f_fd2),1)
        plt.loglog(ms, errors_d2f_fd2, '-', label='fd2 d2f error, a =%f'%a)
        plt.loglog(ms,np.exp(b)*ms**a,'k:')
        
        a,b = np.polyfit(np.log(ms),np.log(errors_df_part2q1),1)
        plt.loglog(ms, errors_df_part2q1, '-', label='part2q1 df error, a =%f'%a)
        plt.loglog(ms,np.exp(b)*ms**a,'k:')
        
        a,b = np.polyfit(np.log(ms),np.log(errors_d2f_part2q1),1)
        plt.loglog(ms, errors_d2f_part2q1, '-', label='part2q1 d2f error, a =%f'%a)
        plt.loglog(ms,np.exp(b)*ms**a,'k:')
        
        
        
        plt.xlabel('m')
        plt.ylabel('Error ')
        plt.legend()
        plt.title('Error vs m')
        plt.grid()
        return None
        
    def test3(): #how m and n changinhg at the same time effect the cost and accuracy 
            
            errors_df_fd2 = []
            errors_d2f_fd2 = []
            errors_df_part2q1 = []
            errors_d2f_part2q1 = []
            times_fd2 = []
            times_part2q1 = []
            
            hs = []
            
            mns = []
            total_points = []
            for m,n in zip(np.arange(10,2000,100),np.arange(10,2000,100)):
                x = np.linspace(0,1,n)
                y = np.linspace(0,1,m)  
                X, Y = np.meshgrid(x, y)
                h = 1/(n-1)
                hs.append(h)
                mns.append((m,n))
                total_points.append(m*n)
                
                f = np.sin(np.pi * X) * np.cos(np.pi * Y)
                df_exact = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)   # First derivative with respect to x
                d2f_exact = - (np.pi ** 2) * np.sin(np.pi * X) * np.cos(np.pi * Y)
                
                t1 = time.time()
                df_p1,d2f_p1 = part2q1(f)
                dt1 = time.time() - t1
                times_part2q1.append(dt1)
                
                t2 = time.time()
                df_fd2,d2f_fd2 = fd2(f)
                dt2 = time.time() - t2
                times_fd2.append(dt2)
                
                errordf_fd2 = np.max(np.abs((df_fd2 - df_exact)))
                errors_df_fd2.append(errordf_fd2)
                errord2f_fd2 = np.max(np.abs((d2f_fd2 - d2f_exact)))
                errors_d2f_fd2.append(errord2f_fd2)
                
                errordf_p1 = np.max(np.abs((df_p1 - df_exact)))
                errors_df_part2q1.append(errordf_p1)
                errord2f_p1 = np.max(np.abs((d2f_p1 - d2f_exact)))
                errors_d2f_part2q1.append(errord2f_p1)
                
            total_points = np.array(total_points)
            nskip = int(0.70*len(total_points))
            plt.figure()
            a,b = np.polyfit(np.log(total_points[nskip:]),np.log(times_fd2[nskip:]),1)
            plt.loglog(total_points, times_fd2, '-', label='fd2 time, a=%f'%a)
            plt.loglog(total_points,np.exp(b)*total_points**a,'k:')
            
            a,b = np.polyfit(np.log(total_points[nskip:]),np.log(times_part2q1[nskip:]),1)
            plt.loglog(total_points, times_part2q1, '-', label='part2q1 time, a =%f'%a)
            plt.loglog(total_points,np.exp(b)*total_points**a,'k:')
            
            plt.xlabel('mn ')
            plt.ylabel('Time ')
            plt.legend()
            plt.title('Computation Time vs mn')
            plt.grid()
            
            plt.figure()
            
            a,b = np.polyfit(np.log(total_points),np.log(errors_df_fd2),1)
            plt.loglog(total_points, errors_df_fd2, '-', label='fd2 df error, a =%f'%a)
            plt.loglog(total_points,np.exp(b)*total_points**a,'k:')
            
            a,b = np.polyfit(np.log(total_points),np.log(errors_d2f_fd2),1)
            plt.loglog(total_points, errors_d2f_fd2, '-', label='fd2 d2f error, a =%f'%a)
            plt.loglog(total_points,np.exp(b)*total_points**a,'k:')
            
            a,b = np.polyfit(np.log(total_points),np.log(errors_df_part2q1),1)
            plt.loglog(total_points, errors_df_part2q1, '-', label='part2q1 df error, a =%f'%a)
            plt.loglog(total_points,np.exp(b)*total_points**a,'k:')
            
            a,b = np.polyfit(np.log(total_points),np.log(errors_d2f_part2q1),1)
            plt.loglog(total_points, errors_d2f_part2q1, '-', label='part2q1 d2f error, a =%f'%a)
            plt.loglog(total_points,np.exp(b)*total_points**a,'k:')
            
            plt.xlabel('mn')
            plt.ylabel('Error ')
            plt.legend()
            plt.title('Error vs mn')
            plt.grid()
            
            
            return None 
    def test4():#how m and n changing at different rates effect the cost and accuracy not included in report
            
            from mpl_toolkits.mplot3d import Axes3D
                    
            
            
            hs = []
            ms = np.arange(10,2000,100)
            
            ns =np.arange(10,2000,100)
            
            
            num_n = len(ns)
            num_m = len(ms)
           
            
            errors_df_fd2 = np.zeros((num_m, num_n))
            errors_d2f_fd2 = np.zeros((num_m, num_n))
            errors_df_part2q1 = np.zeros((num_m, num_n))
            errors_d2f_part2q1 = np.zeros((num_m, num_n))
            times_fd2 = np.zeros((num_m, num_n))
            times_part2q1 = np.zeros((num_m, num_n))
           
            for i,m in enumerate(ms):
                for j,n in enumerate(ns):
                    x = np.linspace(0,1,n)
                    y = np.linspace(0,1,m)  
                    X, Y = np.meshgrid(x, y)
                    h = 1/(n-1)
                    hs.append(h)
                    
                    
                    
                    f = np.sin(np.pi * X) * np.cos(np.pi * Y)
                    df_exact = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)   # First derivative with respect to x
                    d2f_exact = - (np.pi ** 2) * np.sin(np.pi * X) * np.cos(np.pi * Y)
                    
                    t1 = time.time()
                    df_p1,d2f_p1 = part2q1(f)
                    dt1 = time.time() - t1
                    times_part2q1[i,j] = dt1
                    
                    t2 = time.time()
                    df_fd2,d2f_fd2 = fd2(f)
                    dt2 = time.time() - t2
                    times_fd2[i,j]= dt2
                    
                    errordf_fd2 = np.max(np.abs((df_fd2 - df_exact)))
                    errors_df_fd2[i,j] =errordf_fd2
                    errord2f_fd2 = np.max(np.abs((d2f_fd2 - d2f_exact)))
                    errors_d2f_fd2[i,j] = errord2f_fd2
                    
                    errordf_p1 = np.max(np.abs((df_p1 - df_exact)))
                    errors_df_part2q1[i,j] = errordf_p1
                    errord2f_p1 = np.max(np.abs((d2f_p1 - d2f_exact)))
                    errors_d2f_part2q1[i,j] = errord2f_p1
                
           
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            plt.imshow(np.log(times_fd2),cmap='viridis', aspect='auto', origin='lower',extent=[min(np.log(ns)), max(np.log(ns)), min(np.log(ms)), max(np.log(ns))] )
            plt.colorbar(label='Time ')
            plt.xlabel('n ')
            plt.ylabel('m ')
            plt.title('heatmap times fd2')
            plt.xticks(np.log(ns[::2]))
            plt.yticks(np.log(ms[::2]))
            
            
            plt.subplot(1,2,2)
            plt.imshow(np.log(times_part2q1),cmap='inferno', aspect='auto', origin='lower',extent=[min(np.log(ns)), max(np.log(ns)), min(np.log(ms)), max(np.log(ns))] )
            plt.colorbar(label='Time ')
            plt.xlabel('n ')
            plt.ylabel('m ')
            plt.title('heatmap times part2q1')
            plt.xticks(np.log(ns[::2]))
            plt.yticks(np.log(ms[::2]))
            plt.show()
            
            N,M = np.meshgrid(ns,ms)
            fig = plt.figure(figsize = (10,10))
            ax=fig.add_subplot(111,projection='3d')
            
            ax.plot_surface(np.log(N), np.log(M), np.log(times_fd2),color='red',edgecolor='red',lw=0.5, rstride=4, cstride=4,alpha = 0.5, label = 'fd2')
            ax.plot_surface(np.log(N), np.log(M), np.log(times_part2q1),color='royalblue',edgecolor='royalblue', lw=0.5, rstride=4, cstride=4, alpha = 0.7, label = 'part2q1')
           
            ax.set_xlabel('log n')
            ax.set_ylabel('log m')
            ax.set_zlabel('log Time (seconds)')
            ax.set_title('Time Surface Plot')
            ax.view_init(elev=10, azim=-45)
            
            ax.legend()
            plt.show()
            
        
            
            return None
        
        
    def wavenumber():
        
            
    
            m = 1000  
            n = 1000  
            x = np.linspace(0,1, n)  
            y = np.linspace(0, 1, m)
            X, Y = np.meshgrid(x, y)
            h = 1/(n-1)  
        
            
            ks = np.linspace(0.1, np.pi / h, 1000)  
        
            fd2_khnew = []
            part2q1_khnew = []
            khs = []
            errors_fd2 =[]
            errors_p2 = []
        
            for k in ks:
                f = np.sin(k * X)
                df_exact = k * np.cos(k * X)  
        
                
                df_p1,d2f_p1 = part2q1(f)
                df_fd2,d2f_fd2 = fd2(f)
        
                
                cos_kX = np.cos(k * X)
                mask = np.abs(cos_kX) > 0.1 #avoiding horizontal boundaries
        
                kh_fd2 =np.mean( df_fd2[mask] / cos_kX[mask])
                kh_part2q1 = np.mean(df_p1[mask] / cos_kX[mask])
        
                
                kh = k * h
                
        
                fd2_khnew.append(kh_fd2 * h)
                part2q1_khnew.append(kh_part2q1 * h)
                khs.append(kh)
        
            
            fd2_khnew = np.array(fd2_khnew)
            part2q1_khnew = np.array(part2q1_khnew)
            khs = np.array(khs)
            error_fd2 =  np.abs(fd2_khnew-khs)/khs
            error_p2 = np.abs(part2q1_khnew-khs)
            indices_fd2 = np.where(error_fd2 < 0.01)[0]
            indices_part2q1 = np.where(error_p2 < 0.01)[0]

            kh_below_1percent_error_fd2 = khs[indices_fd2]
            kh_below_1percent_error_part2q1 = khs[indices_part2q1]
            max_kh_fd2 = np.max(kh_below_1percent_error_fd2) if len(kh_below_1percent_error_fd2) > 0 else None
            max_kh_part2q1 = np.max(kh_below_1percent_error_part2q1) if len(kh_below_1percent_error_part2q1) > 0 else None

            print(f" fd2, the error is less than 1% up to kh ≈ {max_kh_fd2:.3f}")
            print(f" part2q1, the error is less than 1% up to kh ≈ {max_kh_part2q1:.3f}")
        
            
            plt.figure(figsize=(10, 6))
            plt.plot(khs, khs, 'k-', label='Exact ')
            plt.plot(khs, fd2_khnew, 'b', label='fd2')
            plt.plot(khs, part2q1_khnew, 'r', label='part2q1')
            plt.xlabel('$k h$')
            plt.ylabel('kh modified')
            plt.title('Modified Wavenumber vs. Exact Wavenumber')
            plt.legend()
            plt.grid()
            plt.show()
            return None
                
        
       
                  
            
            
    return   wavenumber() ,test1(),test2(),test3(),test4()   


#---------------------------- End code for Part 2 ----------------------------#

if __name__=='__main__':
    x=0 #please do not remove
    #Add code here to call functions used to generate the figures included in your report.
