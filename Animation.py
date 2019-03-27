import vpython as vp
import numpy as np 
%matplotlib inline
import matplotlib.pyplot as plt 
import numpy as np 
G=6.67*1e-11
M1=15*1e27
M2=15*1e27
M3=30*1e27
rini1=np.array([0,0,0, 0,np.sqrt(5*G*M2/4/1e11),-np.sqrt(5*G*M2/4/1e11), 0,1e11,-1e11, 0,0,0, 0,0,0, 0,0,0]) 
rini2=np.array([0,0,0, np.sqrt(G*M1/4e11),-np.sqrt(G*M1/4e11),np.sqrt(2*G*M1/30e11), 1e11,-1e11,30e11, 0,0,0, 0,0,0, 0,0,0])
rini3=np.array([0.5*1e11,-0.5*1e11,0, -(np.sqrt(3)/np.sqrt(7))*np.sqrt(7*G*M1/4e11),-(np.sqrt(3)/np.sqrt(7))*np.sqrt(7*G*M1/4e11),np.sqrt(3*G*M1/4e11), 0,0,0.5*1e11*np.sqrt(3), -(2/np.sqrt(7))*np.sqrt(7*G*M1/4e11),(2/np.sqrt(7))*np.sqrt(7*G*M1/4e11),0, 0,0,0, 0,0,0])
ball1 = vp.sphere(pos=vp.vector(rini1[0],rini1[6],rini1[12]), radius=4000000000,make_trail=True, interval=1, retain=200, trail_color=vp.color.green)
ball2 = vp.sphere(pos=vp.vector(rini1[1],rini1[7],rini1[13]), radius=4000000000,make_trail=True, interval=1, retain=200, trail_color=vp.color.red)
ball3 = vp.sphere(pos=vp.vector(rini1[2],rini1[8],rini1[14]), radius=4000000000,make_trail=True, interval=1, retain=200, trail_color=vp.color.blue)

def f(r,t):
    x1=r[0]
    x2=r[1]
    x3=r[2]
    vx1=r[3]
    vx2=r[4]
    vx3=r[5]
    y1=r[6]
    y2=r[7]
    y3=r[8]
    vy1=r[9]
    vy2=r[10]
    vy3=r[11]
    z1=r[12]
    z2=r[13]
    z3=r[14]
    vz1=r[15]
    vz2=r[16]
    vz3=r[17] 
    r12=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2) 
    r13=np.sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2) 
    r23=np.sqrt((x3-x2)**2+(y3-y2)**2+(z3-z2)**2) 
    fx1=vx1
    fy1=vy1
    fz1=vz1
    fx2=vx2
    fy2=vy2
    fz2=vz2
    fx3=vx3
    fy3=vy3
    fz3=vz3 
    fvx1=G*M2*(x2-x1)/abs(r12)**3+G*M3*(x3-x1)/abs(r13)**3 
    fvy1=G*M2*(y2-y1)/abs(r12)**3+G*M3*(y3-y1)/abs(r13)**3 
    fvz1=G*M2*(z2-z1)/abs(r12)**3+G*M3*(z3-z1)/abs(r13)**3 
    fvx2=G*M1*(x1-x2)/abs(r12)**3+G*M3*(x3-x2)/abs(r23)**3 
    fvy2=G*M1*(y1-y2)/abs(r12)**3+G*M3*(y3-y2)/abs(r23)**3 
    fvz2=G*M1*(z1-z2)/abs(r12)**3+G*M3*(z3-z2)/abs(r23)**3 
    fvx3=G*M2*(x2-x3)/abs(r23)**3+G*M1*(x1-x3)/abs(r13)**3 
    fvy3=G*M2*(y2-y3)/abs(r23)**3+G*M1*(y1-y3)/abs(r13)**3
    fvz3=G*M2*(z2-z3)/abs(r23)**3+G*M1*(z1-z3)/abs(r13)**3
    return np.array([fx1,fx2,fx3,fvx1,fvx2,fvx3,fy1,fy2,fy3,fvy1,fvy2,fvy3,fz1,fz2,fz3,fvz1,fvz2,fvz3],dtype=np.float) 

def RK4(f, r, tleft, tright, dt, err_tol=1000/31556952):
    def rk4_update(r, t, dt): 
        k1 = dt * f(r, t)
        k2 = dt * f(r+0.5*k1, t+0.5*dt)
        k3 = dt * f(r+0.5*k2, t+0.5*dt)
        k4 = dt * f(r+k3, t+dt)
        return r + (k1 + 2*k2 + 2*k3 + k4)/6.0
    t=tleft 
    x1points=[] 
    y1points=[] 
    z1points=[]
    x2points=[] 
    y2points=[] 
    z2points=[] 
    x3points=[] 
    y3points=[] 
    z3points=[] 
    while t <= tright:
        x1points.append(r[0]) 
        x2points.append(r[1]) 
        x3points.append(r[2]) 
        y1points.append(r[6]) 
        y2points.append(r[7]) 
        y3points.append(r[8]) 
        z1points.append(r[12]) 
        z2points.append(r[13]) 
        z3points.append(r[14]) 
        while True:
            r1_a=rk4_update(r, t, dt).copy()
            r1=rk4_update(r1_a, t+dt, dt).copy()
            r2=rk4_update(r, t, 2*dt).copy()
            eps_x1 = np.abs(r1[0] - r2[0])/30.0
            eps_y1 = np.abs(r1[6] - r2[6])/30.0
            eps_z1 = np.abs(r1[12] - r2[12])/30.0
            eps_tot1 = np.sqrt(eps_x1**2 + eps_y1**2+eps_z1**2) 
            eps_x2 = np.abs(r1[1] - r2[1])/30.0
            eps_y2 = np.abs(r1[7] - r2[7])/30.0
            eps_z2= np.abs(r1[13] - r2[13])/30.0
            eps_tot2 = np.sqrt(eps_x2**2 + eps_y2**2+eps_z2**2) 
            eps_x3 = np.abs(r1[2] - r2[2])/30.0
            eps_y3 = np.abs(r1[8] - r2[8])/30.0
            eps_z3 = np.abs(r1[14] - r2[14])/30.0

            eps_tot3 = np.sqrt(eps_x3**2 + eps_y3**2+eps_z3**2)
            rho = min((dt*err_tol/eps_tot1)**(1./4),(dt*err_tol/eps_tot2)**(1./4),(dt*err_tol/eps_tot3)**(1./4)) 
            if rho >= 1.0:
                if rho >= 2.0:
                  rho = 2.0
                break 
            else:
                if rho < 0.5:
                  rho = 0.5 
                dt *= 0.99*rho
        dt *= 0.99*rho
        r = r1_a.copy() 
        vp.rate(20000000/dt)
        t += dt 
        ball1.pos=vp.vector(r[0],r[6],r[12]) 
        ball2.pos=vp.vector(r[1],r[7],r[13]) 
        ball3.pos=vp.vector(r[2],r[8],r[14])
    x1points = np.array(x1points)
    y1points = np.array(y1points)
    x2points = np.array(x2points)
    y2points = np.array(y2points)
    x3points = np.array(x3points)
    y3points = np.array(y3points)
    return x1points,y1points,z1points,x2points,y2points,z2points,x3points,y3points,z3points
tleft = 0.0
tright = 400000000.0
N = 50000
dt =float(tright- tleft) / N 
xp1,yp1,zp1,xp2,yp2,zp2,xp3,yp3,zp3=RK4(f,rini3,tleft,tright,dt,err_tol=1000/31556952) 
fig = plt.figure(figsize=(20,20))
ax2 = fig.add_subplot(1, 1, 1)
ax2.set_xlabel('x', fontsize=20)
ax2.set_ylabel('y', fontsize=18)
ax2.set_xlim(-0.5*1e+12,0.5*1e+12)
ax2.set_ylim(-0.5*1e+12,0.5*1e+12)
ax2.plot(xp2, yp2)
ax2.plot(xp1, yp1)
ax2.plot(xp3, yp3)
