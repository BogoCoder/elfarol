import pandas as pd
import seaborn as sns
from random import uniform
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import minimize, Bounds
from itertools import product

class Agente :
    
    def __init__(self, states, puntajes, atractivos) :
        self.state = states # lista
        self.puntaje = puntajes # lista
        self.attendance_p = atractivos # lista
        
    def toma_decision(self, modelo, parametros, threshold, attendances, DEB=False) :
        '''
        Se lanza un "dado" para decidir aleatoriamente si el agente va o no al bar
        de acuerdo a si supera el umbral dado por el valor de para_ir
        '''
        attendance_p=self.attendance_p[-1] # Valor por defecto
        beta=0

        # determina el valor de para_ir de acuerdo al modelo y sus parámetros
        if modelo == 'aleatorio' :
            attendance_p=parametros[0]
        
        if modelo == 'belletal' :
            N = threshold
            Nk = np.sum(attendances)
            xk = self.state[-1]
            mu=parametros[0]
            attendance_p=self.attendance_p[-1]-mu*(Nk-N)*xk
            attendance_p = 1 if attendance_p > 1 else attendance_p
            attendance_p = 0 if attendance_p < 0 else attendance_p

        if modelo == 'win-stay-lose-shift' :
            beta=parametros[0]
            epsilon = parametros[1]
            gamma = parametros[2]
            if self.puntaje[-1] >= gamma :
#                attendance_p = 1 - epsilon/2
                attendance_p = max(self.attendance_p[-1] - epsilon/2, 0)
            else:
#                attendance_p = epsilon/2
                attendance_p = max((1 - self.attendance_p[-1]) - epsilon/2, 0)

        # Actualiza el attendance_p del agente
        self.attendance_p.append(attendance_p)

        
        # Lanza el dado
        if uniform(0,1) < attendance_p :
            self.state.append(1)
        else :
            self.state.append(0)

    def imprime_agente(self, ronda) :
        try:
            state = self.state[ronda]
        except:
            state = "nan"
        try:
            puntaje = self.puntaje[ronda]
        except:
            puntaje = "nan"
        try:
            attendance_p = self.attendance_p[ronda]
        except:
            attendance_p = "nan"

        return "state:{0}, Puntaje:{1}, attendance_p:{2}".format(state, puntaje, attendance_p)

from random import randint, uniform 
import numpy as np

class BarElFarol :
    
    def __init__(self, num_agentes, umbral, modelo='aleatorio', parametros=[.5]) :
        self.num_agentes = num_agentes
        self.umbral = umbral
        self.historia = []
        self.agentes = []
        for i in range(self.num_agentes) :
            if modelo == 'aleatorio' :
                self.agentes.append(Agente([randint(0,1)], [], [parametros[0]]))        
            if modelo == 'belletal' :
                self.agentes.append(Agente([randint(0,1)], [], [uniform(0,1)]))  
            if modelo == 'rescorla-wagner' :
                self.agentes.append(Agente([randint(0,1)], [], [parametros[1]]))        
            if modelo == 'win-stay-lose-shift' :
#                self.agentes.append(Agente([randint(0,1)], [], uniform(0,1))) 
                aleatorio = randint(0,1)
                self.agentes.append(Agente([aleatorio], [], [aleatorio])) 
            
    def calcular_asistencia(self) :
        asistencia = np.sum([a.state[-1] for a in self.agentes])
        self.historia.append(asistencia)

    def calcular_puntajes(self) :
        asistencia = self.historia[-1]/self.num_agentes
        for a in self.agentes:
            if a.state[-1] == 1:
                if asistencia > self.umbral:
                    a.puntaje.append(-1)
                else:
                    a.puntaje.append(1)
            else:
                a.puntaje.append(0)

    def agentes_deciden(self, modelo='aleatorio', parametros=[0.5], DEB=False) :
        for i, a in enumerate(self.agentes) :
            attendances = [a.state[-1]] + [self.agentes[j].state[-1] for j in range(self.num_agentes) if j != i]
            a.toma_decision(modelo, parametros, threshold=self.umbral*self.num_agentes, attendances=attendances, DEB=DEB)
                
    def imprime_ronda(self, ronda) :
        try:
            asistencia = self.historia[ronda]
        except:
            asistencia = "nan"
        cadena = '='*30
        cadena += f"\nRonda: {ronda} || Asistencia: {asistencia}"
        for a in self.agentes:
            cadena += "\n" + a.imprime_agente(ronda)
        print(cadena)
        
    def guardar_pandas(self) :
        dict = {}
        ronda = []
        agentes = []
        states = []
        puntajes = []
        atractivos = []
        for i in range(len(self.agentes)):
            a = self.agentes[i]
            ronda += [x for x in range(1, len(a.state))]
            agentes += [i]*len(a.puntaje)
            states += a.state[:-1]
            puntajes += a.puntaje
            atractivos += a.attendance_p[:-1]
#        print(len(agentes), len(states), len(puntajes), len(atractivos))
        dict['round'] = ronda
        dict['player'] = agentes
        dict['choice'] = states
        dict['score'] = puntajes
        dict['motivated'] = atractivos
        return pd.DataFrame.from_dict(dict) 

def simulacion(modelo,parametros,num_agentes,umbral,num_iteraciones,Nsim=1) :
    lista_dataframes = []
    for n in range(Nsim):
        bar=BarElFarol(
            num_agentes, 
            umbral, 
            num_iteraciones, 
            modelo=modelo, 
            parametros=parametros
        )        
        for t in range(num_iteraciones) : 
            bar.calcular_asistencia()
            bar.calcular_puntajes()
            bar.agentes_deciden(modelo=modelo,parametros=parametros,DEB=False)
        data = bar.guardar_pandas()
        data.sort_values(by=['round', 'player'], inplace=True)
        data['group'] = n
        data=data[['group','round','player','choice','score','motivated']]
        lista_dataframes.append(data)
    return pd.concat(lista_dataframes)

def visual(data_sim, data_obs) :
    data = pd.concat([data_sim,data_obs])
    attendance = pd.DataFrame(data.groupby(['source', 'group', 'round'])['choice'].sum().reset_index())
    attendance.columns = ['source', 'group', 'round', 'attendance']
    scores = pd.DataFrame(data.groupby(['source','group','player'])['score'].mean().reset_index())
    fig,ax=plt.subplots(1,2,figsize=(8,4))
    sns.lineplot('round', 'attendance', hue='source', data=attendance,ax=ax[0])
    sns.boxplot(data=scores, y='score', x='source', ax=ax[1])
    ax[0].set_title("Average attendance per round")
    ax[1].set_title("Distribution of average score")