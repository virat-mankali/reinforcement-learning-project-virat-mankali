"""Revision of the supplied RL_Inventory_Management_FINAL notebook.
Custom single-store simulator; NOT or-gym InvManagement-v1.
"""
from __future__ import annotations
import json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class InventoryEnv(gym.Env):
    """30-day lost-sales problem with capacity reserved for ALL pipeline stock.

    Decision -> arrivals due today -> demand -> holding charge. An order on day
    t arrives before demand on day t+2. Integer quantities, five-unit action grid.
    Opening inventory is purchased; all ending on-hand/pipeline stock is credited
    at unit cost. This explicit salvage assumption prevents unpaid opening stock
    or hidden terminal purchasing bias; it is not a real retail cash-flow model.
    """
    metadata = {"render_modes": []}
    def __init__(self, penalty_multiplier=1., horizon=30, lead_time=2,
                 capacity=250, max_order=120, demand_mean=20., initial_inventory=40,
                 price=100., unit_cost=60., holding_cost=2., stockout_penalty=50.):
        super().__init__()
        for name,value in [('horizon',horizon),('lead_time',lead_time),('capacity',capacity),('max_order',max_order)]:
            if not isinstance(value,(int,np.integer)) or value<=0: raise ValueError(f'{name} must be a positive integer')
        if not isinstance(initial_inventory,(int,np.integer)) or not 0<=initial_inventory<=capacity:
            raise ValueError('initial_inventory must be an integer within capacity')
        if max_order%5: raise ValueError('max_order must be divisible by action grid size 5')
        if any(not np.isfinite(x) or x<0 for x in [penalty_multiplier,demand_mean,price,unit_cost,holding_cost,stockout_penalty]):
            raise ValueError('Economic and demand parameters must be finite and non-negative')
        self.penalty_multiplier=float(penalty_multiplier)
        self.horizon=horizon; self.lead_time=lead_time; self.capacity=capacity
        self.max_order=max_order; self.demand_mean=demand_mean
        self.initial_inventory=initial_inventory; self.price=price
        self.unit_cost=unit_cost; self.holding_cost=holding_cost
        self.stockout_penalty=stockout_penalty
        self.quantities=np.arange(0,max_order+1,5,dtype=int)
        self.action_space=spaces.Discrete(len(self.quantities))
        # on-hand, each lead-time bucket, time remaining. IID demand history
        # is intentionally omitted: it does not forecast tomorrow's distribution.
        self.observation_space=spaces.Box(0.,1.,shape=(lead_time+2,),dtype=np.float32)
    def _obs(self):
        return np.array([self.on_hand/self.capacity,*[q/self.capacity for q in self.pipeline],
                         (self.horizon-self.day)/self.horizon],dtype=np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.on_hand=self.initial_inventory; self.pipeline=[0]*self.lead_time; self.day=0
        self.demands=None
        if options and 'demands' in options:
            raw=np.asarray(options['demands'])
            if raw.shape!=(self.horizon,) or not np.issubdtype(raw.dtype,np.number) or not np.all(np.isfinite(raw)) or np.any(raw<0) or np.any(raw!=np.floor(raw)):
                raise ValueError('Demand scenario must have horizon non-negative integer values')
            self.demands=raw.astype(int)
        return self._obs(),{}
    def step(self, action):
        if self.day>=self.horizon: raise RuntimeError('Call reset after episode end')
        if not self.action_space.contains(action): raise ValueError('Invalid discrete action')
        requested=int(self.quantities[action])
        # Capacity reservation: no accepted inventory ever disappears at arrival.
        accepted=min(requested,self.capacity-self.on_hand-sum(self.pipeline))
        arriving=self.pipeline.pop(0)
        self.on_hand+=arriving; self.pipeline.append(accepted)
        demand=int(self.np_random.poisson(self.demand_mean)) if self.demands is None else int(self.demands[self.day])
        sold=min(self.on_hand,demand); lost=demand-sold; self.on_hand-=sold
        revenue=self.price*sold; purchase=self.unit_cost*accepted
        opening=self.unit_cost*self.initial_inventory if self.day==0 else 0.
        holding=self.holding_cost*self.on_hand; stockout=self.stockout_penalty*lost
        self.day+=1; terminated=self.day==self.horizon
        salvage=self.unit_cost*(self.on_hand+sum(self.pipeline)) if terminated else 0.
        business_profit=revenue-purchase-opening-holding-stockout+salvage
        shaped_profit=business_profit-(self.penalty_multiplier-1)*stockout
        info=dict(day=self.day,demand=demand,sold=sold,lost_units=lost,on_hand=self.on_hand,
                  in_transit=sum(self.pipeline),arrivals=arriving,requested_order=requested,
                  accepted_order=accepted,rejected_order=requested-accepted,revenue=revenue,
                  purchase_cost=purchase,opening_inventory_cost=opening,holding_cost=holding,
                  stockout_cost=stockout,terminal_salvage=salvage,business_profit=business_profit,
                  shaped_profit=shaped_profit,stockout_day=int(lost>0))
        return self._obs(),shaped_profit/1000.,terminated,False,info

def scenarios(seed,n=100,stress=False,horizon=30):
    # Each episode has its own independent RNG stream; seed banks never overlap.
    children=np.random.SeedSequence(seed).spawn(n)
    rates=np.full(horizon,20.)
    if stress: rates[14:17]*=3 # days 15-17, no warning in the observation
    return np.stack([np.random.default_rng(s).poisson(rates) for s in children])

def heuristic(s,S):
    def policy(obs,env):
        position=env.on_hand+sum(env.pipeline)
        target=min(env.max_order,max(0,S-position)) if position<s else 0
        return int(np.argmin(abs(env.quantities-target)))
    return policy

def evaluate(policy,bank,policy_name,training_seed=-1,scenario='normal'):
    env=InventoryEnv(); rows=[]; trajectories=[]
    for episode,demands in enumerate(bank):
        obs,_=env.reset(options={'demands':demands}); daily=[]
        while True:
            action=policy(obs,env); obs,_,done,_,info=env.step(action); daily.append(info)
            if done: break
        df=pd.DataFrame(daily)
        row={k:float(df[k].sum()) for k in ['business_profit','holding_cost','stockout_cost','lost_units','demand','sold','stockout_day','revenue','purchase_cost','opening_inventory_cost','terminal_salvage']}
        row.update(policy=policy_name,training_seed=training_seed,scenario=scenario,episode=episode,
                   average_daily_profit=row['business_profit']/env.horizon,
                   average_daily_holding=row['holding_cost']/env.horizon,
                   fill_rate=row['sold']/row['demand'] if row['demand'] else 1.,
                   stockout_day_rate=row['stockout_day']/env.horizon,
                   ending_on_hand=env.on_hand,ending_pipeline=sum(env.pipeline))
        rows.append(row)
        if episode==0:
            df['policy']=policy_name; df['training_seed']=training_seed; df['scenario']=scenario
            trajectories.append(df)
    return pd.DataFrame(rows),pd.concat(trajectories,ignore_index=True)

def model_policy(model):
    return lambda obs,env: int(model.predict(obs,deterministic=True)[0])

def make_model(seed,penalty_multiplier,log_dir):
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    env=Monitor(InventoryEnv(penalty_multiplier=penalty_multiplier),str(log_dir),
                info_keywords=('business_profit',))
    return DQN('MlpPolicy',env,learning_rate=3e-4,buffer_size=50000,
               learning_starts=3000,batch_size=128,gamma=1.0,train_freq=4,
               gradient_steps=1,target_update_interval=1000,exploration_fraction=.5,
               exploration_final_eps=.05,policy_kwargs={'net_arch':[64,64]},seed=seed,
               device='cpu',verbose=0)

def train_and_evaluate(output='artifacts',steps=150_000,seeds=(11,22,33)):
    """Train from scratch; tune only on validation; then unlock fixed final test banks."""
    import torch, stable_baselines3, gymnasium, matplotlib
    from stable_baselines3.common.callbacks import BaseCallback
    out=Path(output); out.mkdir(exist_ok=True,parents=True)
    for p in ['models','logs','metrics','figures']: (out/p).mkdir(exist_ok=True)
    val=scenarios(10100,50)
    grid=[]
    for s in range(30,101,10):
        for S in range(max(50,s),141,10):
            d,_=evaluate(heuristic(s,S),val,'heuristic')
            grid.append(dict(s=s,S=S,validation_profit=d.business_profit.mean()))
    grid=pd.DataFrame(grid).sort_values('validation_profit',ascending=False)
    grid.to_csv(out/'metrics/heuristic_validation.csv',index=False)
    chosen=grid.iloc[0]; best_s,best_S=int(chosen.s),int(chosen.S)
    print(f'Validation-selected heuristic: s={best_s}, S={best_S}',flush=True)
    training_rows=[]; validation_rows=[]; timing=[]; models={}
    class Diagnostics(BaseCallback):
        def __init__(self,label,seed):
            super().__init__(); self.label=label; self.seed=seed; self.ep=[]; self.best=-np.inf
        def _on_step(self):
            info=self.locals['infos'][0]; self.ep.append(info)
            if self.locals['dones'][0]:
                training_rows.append(dict(policy=self.label,training_seed=self.seed,step=self.num_timesteps,
                    business_profit=sum(x['business_profit'] for x in self.ep),
                    shaped_profit=sum(x['shaped_profit'] for x in self.ep),
                    lost_units=sum(x['lost_units'] for x in self.ep),
                    mean_order=np.mean([x['accepted_order'] for x in self.ep]),
                    epsilon=self.model.exploration_rate))
                self.ep=[]
            if self.num_timesteps % 25000==0 or self.num_timesteps==steps:
                d,_=evaluate(model_policy(self.model),val,self.label,self.seed)
                score=float(d.business_profit.mean())
                validation_rows.append(dict(policy=self.label,training_seed=self.seed,step=self.num_timesteps,
                    business_profit=score,fill_rate=float(d.fill_rate.mean()),lost_units=float(d.lost_units.mean())))
                if score>self.best:
                    self.best=score; self.model.save(out/f'models/{self.label}_seed{self.seed}_best.zip')
                print(f'{self.label} seed {self.seed}: {self.num_timesteps:,} steps, validation profit {score:,.1f}',flush=True)
            return True
    from stable_baselines3 import DQN
    for seed in seeds:
        for label,mult in [('baseline_dqn',1.),('penalty10_dqn',10.)]:
            start=time.perf_counter(); model=make_model(seed,mult,out/f'logs/{label}_{seed}')
            callback=Diagnostics(label,seed); model.learn(total_timesteps=steps,callback=callback)
            model.save(out/f'models/{label}_seed{seed}_final.zip')
            models[(label,seed)]=DQN.load(out/f'models/{label}_seed{seed}_best.zip',device='cpu')
            timing.append(dict(policy=label,training_seed=seed,steps=steps,seconds=time.perf_counter()-start))
            pd.DataFrame(training_rows).to_csv(out/'metrics/training_episodes.csv',index=False)
            pd.DataFrame(validation_rows).to_csv(out/'metrics/validation_checkpoints.csv',index=False)
    # Test data never participates in hyperparameter/checkpoint/heuristic selection.
    all_metrics=[]; all_trajectories=[]
    for scenario,bank in [('normal',scenarios(20200,100)),('stress',scenarios(30300,100,True))]:
        pd.DataFrame(bank).to_csv(out/f'metrics/{scenario}_demand_bank.csv',index=False)
        for label,policy,seed in [('sS_heuristic',heuristic(best_s,best_S),-1),
                                  *[(k[0],model_policy(v),k[1]) for k,v in models.items()]]:
            d,t=evaluate(policy,bank,label,seed,scenario)
            all_metrics.append(d); all_trajectories.append(t)
    metrics=pd.concat(all_metrics,ignore_index=True); traj=pd.concat(all_trajectories,ignore_index=True)
    metrics.to_csv(out/'metrics/test_episodes.csv',index=False)
    traj.to_csv(out/'metrics/representative_trajectories.csv',index=False)
    pd.DataFrame(val).to_csv(out/'metrics/validation_demand_bank.csv',index=False)
    summary=metrics.groupby(['scenario','policy']).agg(
        monthly_profit=('business_profit','mean'),daily_profit=('average_daily_profit','mean'),
        monthly_holding=('holding_cost','mean'),lost_units=('lost_units','mean'),fill_rate=('fill_rate','mean'),
        stockout_days=('stockout_day','mean'),stockout_day_rate=('stockout_day_rate','mean')).reset_index()
    summary.to_csv(out/'metrics/summary.csv',index=False)
    per_seed=metrics.groupby(['scenario','policy','training_seed']).agg(
        profit=('business_profit','mean'),fill_rate=('fill_rate','mean'),lost_units=('lost_units','mean')).reset_index()
    per_seed.to_csv(out/'metrics/per_seed_summary.csv',index=False)
    # Pair by demand episode, average RL seeds within episode first. The interval
    # measures demand uncertainty CONDITIONAL on these trained models, not seed uncertainty.
    paired=[]; rng=np.random.default_rng(40400)
    for scenario in ['normal','stress']:
        sub=metrics[metrics.scenario==scenario]
        h=sub[sub.policy=='sS_heuristic'].set_index('episode').business_profit.sort_index()
        for label in ['baseline_dqn','penalty10_dqn']:
            r=sub[sub.policy==label].groupby('episode').business_profit.mean().sort_index()
            diff=(r-h).to_numpy(); boot=rng.choice(diff,(5000,len(diff)),replace=True).mean(axis=1)
            paired.append(dict(scenario=scenario,comparison=f'{label} minus heuristic',profit_difference=diff.mean(),
                               ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975)))
        b=sub[sub.policy=='baseline_dqn'].groupby('episode').business_profit.mean().sort_index()
        p=sub[sub.policy=='penalty10_dqn'].groupby('episode').business_profit.mean().sort_index()
        diff=(p-b).to_numpy(); boot=rng.choice(diff,(5000,len(diff)),replace=True).mean(axis=1)
        paired.append(dict(scenario=scenario,comparison='penalty10_dqn minus baseline_dqn',profit_difference=diff.mean(),
                           ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975)))
    pd.DataFrame(paired).to_csv(out/'metrics/paired_comparisons.csv',index=False)
    manifest=dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
        torch=torch.__version__,gymnasium=gymnasium.__version__,stable_baselines3=stable_baselines3.__version__,
        numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,cuda=torch.cuda.is_available(),
        torch_threads=torch.get_num_threads(),training_steps_per_model=steps,training_seeds=list(seeds),
        heuristic={'s':best_s,'S':best_S},timings=timing,total_training_seconds=sum(x['seconds'] for x in timing),
        validation_bank_seed=10100,normal_test_bank_seed=20200,stress_test_bank_seed=30300,
        note='Custom single-store environment; not or-gym. No test-driven tuning. Test episodes shared across models.')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    return metrics,traj,summary,pd.DataFrame(training_rows),pd.DataFrame(validation_rows)

def plot_results(metrics,traj,summary,training,validation,output='artifacts'):
    import matplotlib.pyplot as plt
    out=Path(output)/'figures'; out.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False})
    colors={'baseline_dqn':'#2866a4','penalty10_dqn':'#cd6524','sS_heuristic':'#31835c'}
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for (label,seed),df in training.groupby(['policy','training_seed']):
        for ax,col,title in zip(axes,['business_profit','shaped_profit'],['Original business profit (training, exploratory)','Shaped objective (different reward definitions)']):
            ax.plot(df.step,df[col].rolling(100,min_periods=100).mean(),color=colors[label],alpha=.65,label=f'{label}, seed {seed}')
            ax.set(title=title,xlabel='Training timesteps',ylabel='30-day total, 100-episode moving average')
    axes[0].legend(fontsize=7); fig.tight_layout();fig.savefig(out/'training_curves.png');plt.show()
    fig,ax=plt.subplots(figsize=(9,4))
    for (label,seed),df in validation.groupby(['policy','training_seed']):
        ax.plot(df.step,df.business_profit,marker='o',label=f'{label}, seed {seed}')
    ax.set(title='Checkpoint validation under original business costs',xlabel='Training timesteps',ylabel='Mean 30-day profit');ax.legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(out/'validation_curves.png');plt.show()
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,scenario in enumerate(['normal','stress']):
        data=summary[summary.scenario==scenario].set_index('policy').reindex(['sS_heuristic','baseline_dqn','penalty10_dqn'])
        for ax,col,title in zip(axes[row],['monthly_profit','monthly_holding','lost_units'],['30-day profit','30-day holding cost','30-day lost units']):
            ax.bar(['(s,S)','DQN','10x DQN'],data[col],color=[colors[x] for x in data.index]);ax.set_title(f'{scenario}: {title}')
    fig.tight_layout();fig.savefig(out/'normal_stress_comparison.png');plt.show()
    # Fixed first test episode and fixed seed 11, not an attractive cherry-picked run.
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    rep=traj[(traj.scenario=='stress') & traj.training_seed.isin([-1,11])]
    for label,df in rep.groupby('policy'):
        for ax,col in zip(axes,['on_hand','in_transit','accepted_order']):
            ax.plot(df.day,df[col],label=label,color=colors[label]);ax.set_ylabel(col.replace('_',' '))
    demand=rep[rep.policy=='sS_heuristic']
    axes[0].plot(demand.day,demand.demand,'k--',label='Demand')
    for ax in axes: ax.axvspan(14.5,17.5,color='red',alpha=.08);ax.legend(fontsize=8)
    axes[0].set_title('Fixed representative stress month: days 15–17 have triple expected demand')
    axes[-1].set_xlabel('Day');fig.tight_layout();fig.savefig(out/'stress_trajectory.png');plt.show()
