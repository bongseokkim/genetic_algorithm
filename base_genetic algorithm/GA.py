import numpy as np 


def objective_function(x):
    """
    input : list [x[0], x[1]]
    output : 
    """
    return 50.001 - x[0]**2.0 - x[1]**2.0 
    
def max_one(x):
    return sum(x)



class GA_Binary(object):
    def __init__(self, objective_fn=None, selection_method=None, n_bits=50, n_pop=100,n_iter=1000):
        self.objective_fn = objective_fn
        self.selection_method = selection_method
        self.n_bits = n_bits
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.r_cross = 0.9 
        self.r_mut = 1/float(n_bits)

    
    def roullet_selection(self, population, scores):
        n = len(population)
        scores = np.array(scores)
        proportion = scores/sum(scores)
        selection_idx = np.random.choice(range(n), size=1, p= proportion)
        return population[int(selection_idx)]

    def tournament_selection(self, population, scores, k=3, replace=True):
        ## need to check agian 
        n = len(population)
        scores = np.array(scores)
        candidate = np.random.choice(range(n), size=k, replace=replace)
        max_ind = np.argmax(scores[candidate])
        return population[candidate[max_ind]]

    def crossover(self, p1,p2, r_cross):
        p1 = np.array(p1)
        p2 = np.array(p2)
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross : 
            pt = np.random.randint(1, len(p1)-2) # 양 끝단 하나는 있어야함
            c1 = np.concatenate((p1[:pt], p2[pt:]))
            c2 = np.concatenate((p2[:pt], p1[pt:]))
        return [list(c1), list(c2)]

    def mutation(self,bitsting, r_mut):
        for i in range(len(bitsting)):
            if np.random.rand() < r_mut : 
                bitsting[i]= 1-bitsting[i] # beacuse it is binary
        return bitsting

    def run(self):
        pop =[np.random.randint(0,2,self.n_bits).tolist() for _ in range(self.n_pop)]
        n_pop = len(pop)
        best, best_eval = 0, self.objective_fn(pop[0])
        for gen in range(self.n_iter) :
            scores = [self.objective_fn(c) for c in pop]
            for i in range(n_pop):
                if scores[i] > best_eval : 
                    best, best_eval = pop[i], scores[i]
                    print(f' gen : {gen}, new_best :{best}, scores:{best_eval}')
            selected = [self.roullet_selection(pop,scores) for _ in range(n_pop)]
            children = list()
            for i in range(0, n_pop, 2):
                p1,p2 = selected[i], selected[i+1]
                for c in self.crossover(p1,p2, self.r_cross):
                    c = self.mutation(c, self.r_mut)
                    children.append(c)

            pop = children

        return best, best_eval
            



class GA_float(object):
    def __init__(self, objective_fn=None, selection_method=None, n_bits=16, n_pop=100,n_iter=1000, bounds=None):
        self.objective_fn = objective_fn
        self.selection_method = selection_method
        self.n_bits = n_bits
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.r_cross = 0.9 
        self.bounds = bounds
        self.r_mut = 1/(float(n_bits)*len(bounds))

    
    def decode(self, bounds, n_bits, bitsting):
        decoded = list()
        largest = 2**n_bits
        for i in range(len(bounds)):
            start, end = i*n_bits, (i*n_bits) +n_bits
            substring = bitsting[start:end]
            chars = ''.join([str(s) for s in substring])
            integer = int(chars, 2)
            value  = bounds[i][0] + (integer/largest) * bounds[i][1]-bounds[i][0]
            decoded.append(value)
        

        return decoded


    
    def roullet_selection(self, population, scores):
        n = len(population)
        scores = np.array(scores)
        proportion = scores/sum(scores)
        selection_idx = np.random.choice(range(n), size=1, p= proportion)
        return population[int(selection_idx)]

    def tournament_selection(self, population, scores, k=3, replace=True):
        ## need to check agian 
        n = len(population)
        scores = np.array(scores)
        candidate = np.random.choice(range(n), size=k, replace=replace)
        max_ind = np.argmax(scores[candidate])
        return population[candidate[max_ind]]

    def crossover(self, p1,p2, r_cross):
        p1 = np.array(p1)
        p2 = np.array(p2)
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross : 
            pt = np.random.randint(1, len(p1)-2) # 양 끝단 하나는 있어야함
            c1 = np.concatenate((p1[:pt], p2[pt:]))
            c2 = np.concatenate((p2[:pt], p1[pt:]))
        return [list(c1), list(c2)]

    def mutation(self,bitsting, r_mut):
        for i in range(len(bitsting)):
            if np.random.rand() < r_mut : 
                bitsting[i]= 1-bitsting[i] # beacuse it is binary
        return bitsting

    def run(self):
        pop =[np.random.randint(0,2,self.n_bits*len(self.bounds)).tolist() for _ in range(self.n_pop)]
        n_pop = len(pop)
        best, best_eval = 0, self.objective_fn(self.decode(self.bounds, self.n_bits, pop[0]))
        for gen in range(self.n_iter) :
            decoded = [self.decode(self.bounds, self.n_bits, p) for p in pop]
            scores = [self.objective_fn(d) for d in decoded]
            for i in range(n_pop):
                if scores[i] > best_eval : 
                    best, best_eval = pop[i], scores[i]
                    print(f' gen : {gen}, new_best :{best}, scores:{best_eval}')
            selected = [self.roullet_selection(pop,scores) for _ in range(n_pop)]
            children = list()
            for i in range(0, n_pop, 2):
                p1,p2 = selected[i], selected[i+1]
                for c in self.crossover(p1,p2, self.r_cross):
                    c = self.mutation(c, self.r_mut)
                    children.append(c)

            pop = children

        return best, best_eval, self.decode(self.bounds,self.n_bits,best)

def main():
    bounds = [[-5.0,5],[-5.0,5]]
    ga= GA_float(objective_fn=objective_function, bounds=bounds,n_bits=16)
    best, best_eval, result = ga.run()
    print(f'best:{best}, score:{best_eval}, decode:{result}')
if __name__ == "__main__":
	main()