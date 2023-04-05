import pickle
from Job_shop import Situation
import numpy as np
from statistics import mean

class Environment:

    def __init__(self):
        self.L = 1

    def Instance_Generator(self, machine, e_ave, job_insert, i):

        filename = './test_instances/'+str(job_insert) + '-' + str(machine) + '-' + str(e_ave) + '/' + 'instance'+ '_' + str(i)

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10]

    def main(self, J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL, at):
        k = 0
        x=[]
        Total_tard=[]
        Total_makespan=[]
        Total_uk_ave=[]
        TR=[]
        for i in range(self.L):
            # Total_reward = 0
            x.append(i+1)
            print('-----------------------start',i+1,'test------------------------------')
            # obs=[0 for i in range(6)]
            # obs = np.expand_dims(obs, 0)
            # done=False
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL)
            for i in range(O_num):
                k+=1
                # print(obs)
                # at,rt=self.Select_action(obs)
                # print(at)
                if at==0:
                    at_trans=Sit.rule_lwkr()
                if at==1:
                    at_trans=Sit.rule_mwkr()
                if at==2:
                    at_trans=Sit.rule_spt()
                if at==3:
                    at_trans=Sit.rule_lpt()
                if at==4:
                    at_trans=Sit.rule_lopnr()
                if at==5:
                    at_trans=Sit.rule_mopnr()
                if at==6:
                    at_trans=Sit.rule_swkr()
                if at==7:
                    at_trans = Sit.rule_slack()
                if at == 8:
                    at_trans = Sit.rule_cr()
                # if at==8:
                #     at_trans = Sit.rule_cr()
                # at_trans=self.act[at]

                print('The', i, 'th operation>>', 'select action:', at, ' ', 'job ', at_trans[0], 'is assigned for machine ', at_trans[1], ' Arrival time ', A[at_trans[0]])
                Sit.scheduling(at_trans)
                # obs_t=Sit.Features()

                # if i==O_num-1:
                #     done=True
                #obs = obs_t
                # obs_t = np.expand_dims(obs_t, 0)
                # obs = np.expand_dims(obs, 0)
                # print(obs,obs_t)
                # if 0 == rt:
                #     r_t = Sit.reward1(obs[0][5], obs[0][4], obs_t[0][5], obs_t[0][4])
                # else:
                #     r_t = Sit.reward2(obs[0][0], obs_t[0][0])
                # self._append((obs,at,r_t,obs_t,rt,done))
                # if k>self.Batch_size:
                    # batch_obs, batch_action, batch_reward, batch_next_obs,done= self.sample()
                    # self.replay()
                # Total_reward+=r_t
                # obs=obs_t
            total_tadiness=0
            makespan=0
            uk_ave=sum(Sit.UK)/M_num
            Job=Sit.Jobs
            E=0
            K=[i for i in range(len(Job))]
            End=[]
            for Ji in range(len(Job)):
                endTime=max(Job[Ji].End)
                makespan=max(makespan,endTime)
                End.append(endTime)
                if max(Job[Ji].End)>D[Ji]:
                    total_tadiness+=abs(max(Job[Ji].End)-D[Ji])
            print('<<<<<<<<<-----------------total_tardiness:',total_tadiness,'------------------->>>>>>>>>>')
            Total_tard.append(total_tadiness)
            print('<<<<<<<<<-----------------uk_ave:', uk_ave, '------------------->>>>>>>>>>')
            Total_uk_ave.append(uk_ave)
            print('<<<<<<<<<-----------------makespan:', makespan, '------------------->>>>>>>>>>')
            Total_makespan.append(makespan)
            # print('<<<<<<<<<-----------------reward:',Total_reward,'------------------->>>>>>>>>>')
            # TR.append(Total_reward)
            # plt.plot(K,End,color='y')
            # plt.plot(K,D,color='r')
            # plt.show()
        # plt.plot(x,Total_tard)
        # plt.show()
        return mean(Total_tard),mean(Total_uk_ave),mean(Total_makespan)

def test(machine, e_ave, job_insert, x):
    d = Environment()
    
    Total_tard = []
    Total_uk_ave = []
    Total_makespan = []

    for i in range(1):
        Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time, EL = d.Instance_Generator(machine, e_ave, job_insert, i)
        t, uk, ms = d.main(J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time, EL, x)
        
        Total_tard.append(t)
        Total_uk_ave.append(uk)
        Total_makespan.append(ms)

    tard_ave=mean(Total_tard)
    uk_ave=mean(Total_uk_ave)
    makespan_ave=mean(Total_makespan)
    
    std1 = np.std(Total_tard)
    std2 = np.std(Total_uk_ave)
    std3 = np.std(Total_makespan)
    print(str(tard_ave)+","+str(std1)+","+str(uk_ave)+","+str(std2)+","+str(makespan_ave)+","+str(std3))

    file = open('./test_results/' + str(machine) + '-' + str(e_ave) + '-' + str(job_insert) + '-' + str(x+1), 'w+', encoding='utf-8')
    file.write(str(tard_ave)+","+str(std1)+","+str(uk_ave)+","+str(std2)+","+str(makespan_ave)+","+str(std3))
    file.write("\n")
    file.flush()
    file.close()


for i in range (6, 9, 1):
    test(5, 30, 10, i)
