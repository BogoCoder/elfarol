import pandas as pd
import seaborn as sns
from random import uniform
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import minimize, Bounds
from random import randint, uniform 
import numpy as np

class Agent :
    
    def __init__(self, states, scores, attendance_probas) :
        self.states = states # list
        self.scores = scores # list
        self.attendance_probas = attendance_probas # list
        
    def take_decision(self, model, params, threshold, attendances, DEB=False) :

        attendance_p=self.attendance_probas[-1] # Default

        if model == 'random' :
            attendance_p=params[0]
        
        if model == 'belletal' :
            N = threshold
            Nk = np.sum(attendances)
            xk = self.states[-1]
            mu=params[0]
            attendance_p=self.attendance_probas[-1]-mu*(Nk-N)*xk
            attendance_p = 1 if attendance_p > 1 else attendance_p
            attendance_p = 0 if attendance_p < 0 else attendance_p

        if model == 'win-stay-lose-shift' :
            epsilon=params[0]
            last_c_t = self.states[-1]
            last_r_t = self.scores[-1]
            if (last_c_t == 1 and last_r_t == 1) or (last_c_t != 1 and last_r_t == -1 or last_r_t == 0):
                attendance_p = max((1 - self.attendance_probas[-1]) - epsilon/2, 0)
            elif (last_c_t != 1 and last_r_t == 1) or (last_c_t == 1 and last_r_t == -1 or last_r_t == 0):
                attendance_p = max(self.attendance_probas[-1] - epsilon/2, 0) 
                

        # Update attendance probabilities.
        self.attendance_probas.append(attendance_p)

        
        # Throw a dice.
        if uniform(0,1) < attendance_p :
            self.states.append(1)
        else :
            self.states.append(0)

    def print_agent(self, ronda) :
        try:
            state = self.states[ronda]
        except:
            state = "nan"

        try:
            score = self.scores[ronda]
        except:
            score = "nan"
            
        try:
            attendance_p = self.attendance_probas[ronda]
        except:
            attendance_p = "nan"

        return "State:{0}, Score:{1}, Attendance probability:{2}".format(state, score, attendance_p)

class BarElFarol :
    
    def __init__(self, num_agents, threshold, model='random', params=[.5]) :
        self.num_agents = num_agents
        self.threshold = threshold
        self.history = []
        self.agents = []
        for i in range(self.num_agents) :
            if model == 'random' :
                self.agents.append(Agent([randint(0,1)], [], [params[0]]))        
            if model == 'belletal' :
                self.agents.append(Agent([randint(0,1)], [], [uniform(0,1)]))        
            if model == 'win-stay-lose-shift' :
                random = randint(0,1)
                self.agents.append(Agent([random], [], [uniform(0,1)])) 
            
    def compute_attendance(self) :
        attendance = np.sum([a.states[-1] for a in self.agents])
        self.history.append(attendance)

    def compute_scores(self) :
        attendance = self.history[-1]/self.num_agents
        for a in self.agents:
            if a.states[-1] == 1:
                if attendance > self.threshold:
                    a.scores.append(-1)
                else:
                    a.scores.append(1)
            else:
                a.scores.append(0)

    def agents_decide(self, model='random', params=[0.5], DEB=False) :
        for i, a in enumerate(self.agents) :
            attendances = [a.states[-1]] + [self.agents[j].states[-1] for j in range(self.num_agents) if j != i]
            a.take_decision(model, params, threshold=self.threshold*self.num_agents, attendances=attendances, DEB=DEB)
                
    def print_round(self, round) :
        try:
            attendance = self.history[round]
        except:
            attendance = "nan"
        string_text = '='*30
        string_text += f"\nRound: {round} || Attendance: {attendance}"
        for a in self.agents:
            string_text += "\n" + a.print_agent(round)
        print(string_text)
        
    def save_pandas(self) :
        dict = {}
        round = []
        agents = []
        states = []
        scores = []
        probabilities = []
        for i in range(len(self.agents)):
            a = self.agents[i]
            round += [x for x in range(1, len(a.states))]
            agents += [i]*len(a.scores)
            states += a.states[:-1]
            scores += a.scores
            probabilities += a.attendance_probas[:-1]
        dict['round'] = round
        dict['agent'] = agents
        dict['choice'] = states
        dict['score'] = scores
        dict['probability'] = probabilities
        return pd.DataFrame.from_dict(dict) 

def simulation(model,params,num_agents,threshold,num_it,Nsim=1) :
    list_df = []
    for n in range(Nsim):
        bar=BarElFarol(
            num_agents, 
            threshold, 
            num_it, 
            model=model, 
            params=params
        )        
        for t in range(num_it) : 
            bar.compute_attendance()
            bar.compute_scores()
            bar.agents_decide(model=model,params=params,DEB=False)
        data = bar.save_pandas()
        data.sort_values(by=['round', 'agent'], inplace=True)
        data['group'] = n
        data=data[['group','round','agent','choice','score','probability']]
        list_df.append(data)
    return pd.concat(list_df)

def visual(data_sim, data_obs) :
    data = pd.concat([data_sim,data_obs])
    attendance = pd.DataFrame(data.groupby(['source', 'group', 'round'])['choice'].sum().reset_index())
    attendance.columns = ['source', 'group', 'round', 'attendance']
    scores = pd.DataFrame(data.groupby(['source','group','agent'])['score'].mean().reset_index())
    fig,ax=plt.subplots(1,2,figsize=(8,4))
    sns.lineplot('round', 'attendance', hue='source', data=attendance,ax=ax[0])
    sns.boxplot(data=scores, y='score', x='source', ax=ax[1])
    ax[0].set_title("Average attendance per round")
    ax[1].set_title("Distribution of average score")