import numpy as np
import matplotlib as ml
import sklearn as sk
import matplotlib.pyplot as plt

#dataset
data = [1, 3, 4, 5, 7, 8, 1.2, 3.1, 5.4, 8.7, 9, 8,20,25,30,33,50, 89, 85, 100, 90, 60, 70, 80]

mu1 = 2
mu2 = 80
sigma1 = 10
sigma2 = 10



def gaussian(mu, sigma, x) :
    coeffient = (1/(np.sqrt(2*np.pi*sigma**2)))
    exponent = -((x - mu)**2/(2*sigma**2))
    probability = coeffient * np.exp(exponent)
    return probability

#Two clusters/bell curves, normalize so they turn sum to 1, turn them into percentages
def E_Step(mu1,mu2,sigma1,sigma2,pi1, pi2,x):
    clusterA = pi1 * gaussian(mu1,sigma1,x)
    clusterB = pi2 * gaussian(mu2,sigma2,x)
    total = clusterA + clusterB
    # normalized % that data point is in cluster A
    cluster_A_Prob = clusterA / total
    cluster_B_Prob = clusterB / total
    return cluster_A_Prob, cluster_B_Prob

def M_Step(data,memberships):
    data = np.array(data)
    memberships = np.array(memberships)
    #New mean, weighted average
    new_mu = np.sum(memberships * data) / np.sum(memberships)
    #New variance, weighted spread
    new_sigma_sq = np.sum(memberships * (data - new_mu)**2) / np.sum(memberships)
    new_sigma = np.sqrt(new_sigma_sq)
    
    #New mixing weight, what fraction of points belong to this cluster
    new_pi = np.sum(memberships) / len(data)
    return new_mu, new_sigma, new_pi
    
def GMM(data, mu1, mu2, sigma1, sigma2,pi1,pi2, iterations=100, tolerance=0.0001):
    
    for i in range(iterations):
        # E-step, calculate the memberships
        memberships_A = []
        memberships_B = []
        #save old values
        old_mu1, old_mu2 = mu1, mu2
        
        for x in data:
            mem_A, mem_B = E_Step(mu1, mu2, sigma1,sigma2,pi1,pi2, x)
            memberships_A.append(mem_A)
            memberships_B.append(mem_B)
        
        # M-step, update the parameters
        mu1, sigma1, pi1 = M_Step(data, memberships_A)
        mu2, sigma2,pi2 = M_Step(data, memberships_B)
        
        print(f"Iteration {i+1}: mu1={mu1:.2f}, mu2={mu2:.2f}, sigma1={sigma1:.2f}, sigma2={sigma2:.2f}")

     #check if converged
        change = abs(mu1 - old_mu1) + abs(mu2 - old_mu2)
        if change < tolerance:
            print(f"Converged after {i+1} iterations!")
            break
    
    return mu1, mu2, sigma1, sigma2, pi1, pi2

def plot_GMM(data, mu1, mu2, sigma1, sigma2, pi1, pi2):
    data = np.array(data)
    
    #Get final memberships for each point
    colors = []
    for x in data:
        mem_A, mem_B = E_Step(mu1, mu2, sigma1, sigma2, pi1, pi2, x)
        #Color based on which cluster it belongs to more
        colors.append('blue' if mem_A > mem_B else 'red')
    
    #Plot data points
    plt.scatter(data, [0]*len(data), c=colors, s=100, zorder=5)
    
    #Plot bell curves
    xp = np.linspace(min(data)-10, max(data)+10, 1000)
    curve_A = pi1 * gaussian(mu1, sigma1, xp)
    curve_B = pi2 * gaussian(mu2, sigma2, xp)
    
    plt.plot(xp, curve_A, 'blue', label=f'Cluster A: μ={mu1:.1f}, σ={sigma1:.1f}')
    plt.plot(xp, curve_B, 'red', label=f'Cluster B: μ={mu2:.1f}, σ={sigma2:.1f}')
    
    plt.legend()
    plt.title('GMM Clustering Result')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.show()


#Starting guesses
mu1 = 2
mu2 = 80
sigma1 = 10
sigma2 = 10
pi1 = 0.5  #starting with 50/50
pi2 = 0.5


final_mu1, final_mu2, final_sigma1, final_sigma2, final_pi1, final_pi2 = GMM(data, mu1, mu2, sigma1, sigma2, pi1, pi2)
plot_GMM(data, final_mu1, final_mu2, final_sigma1, final_sigma2, final_pi1, final_pi2)
