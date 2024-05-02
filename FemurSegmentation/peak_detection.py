import numpy as np
import pandas as pd

def get_persistence(profile):

    default_value = -1

    groups = np.zeros([len(profile), len(profile)]) +default_value
    persistence = np.zeros_like(profile)

    is_emerged = np.zeros_like(profile)
    is_alive = np.ones_like(profile)

    df = pd.DataFrame({"values": profile, "index": range(len(profile))})
    sorted_values = df.sort_values(by='values', ascending=False)

    # We have a step counter for updating the level, a different step counter for updating the groups
    # So, a counter for changing level, a counter for changing point.
    # Probably it's a lot complicated this reasoning, I'm sure there are better ways of doing it 
    t_lvl = 0
    t_grp = 0
    old_v = np.max(profile)
    for v in sorted_values['values'].unique():

        i_lvl = np.where(profile==v)[0]
        for i in i_lvl:

            is_emerged[i] = 1

            if t_grp!=0:
                groups[t_grp,:] = groups[t_grp-1,:]

            if i+1==len(sorted_values):
                right_group = default_value
                left_group = groups[t_grp, i-1]
            elif i==0:
                left_group = default_value
                right_group = groups[t_grp, i+1]
            else:
                left_group = groups[t_grp, i-1]
                right_group = groups[t_grp, i+1]

            if (left_group==default_value)&(right_group==default_value):
                groups[t_grp, i] = str(i)
            elif (left_group==default_value)&(right_group!=default_value):
                groups[t_grp, i] = right_group
                is_alive[i] = 0
            elif (left_group!=default_value)&(right_group==default_value):
                groups[t_grp, i] = left_group
                is_alive[i] = 0    
            else:
                left_pers = np.max(persistence[groups[t_grp,:]==left_group])
                right_pers = np.max(persistence[groups[t_grp,:]==right_group])
                if left_pers>right_pers:
                    is_alive[groups[t_grp,:]==right_group] = 0
                    groups[t_grp, groups[t_grp,:]==right_group] = left_group
                    groups[t_grp, i] = left_group
                    is_alive[i] = 0
                elif right_pers>left_pers:
                    is_alive[groups[t_grp,:]==left_group] = 0
                    groups[t_grp, groups[t_grp,:]==left_group] = right_group
                    groups[t_grp, i] = right_group
                    is_alive[i] = 0
                else:
                    # Se hanno identica persistenza li diamo a quello di sinistra, anche se credo non possano
                    # avere identica persistenza per com'è fatto l'algoritmo...
                    is_alive[groups[t_grp,:]==right_group] = 0
                    groups[t_grp, groups[t_grp,:]==right_group] = left_group
                    groups[t_grp, i] = left_group
                    is_alive[i] = 0

            t_grp += 1

        # before increasing t we develop all the level
        t_lvl += 1

        step =  old_v-v
        persistence += (is_alive*is_emerged)*step
        old_v = v

    return persistence
