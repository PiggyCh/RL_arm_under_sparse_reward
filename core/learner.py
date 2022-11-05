import torch
import time
import numpy as np
from torch.optim import Adam
from core.buffer import replay_buffer
from copy import deepcopy
from arguments import Args
from core.model import Net

env_params = Args.env_params
train_params = Args.train_params
task_params = Args.task_params

n_agents = env_params.n_agents
dim_hand = env_params.dim_hand
device =  train_params.device
batch_size = train_params.batch_size
learner_step = train_params.learner_step
clip_obs = train_params.clip_obs
gamma = train_params.gamma
polyak = train_params.polyak
evalue_interval = train_params.evalue_interval

preproc = lambda norm_func, tensor_data : norm_func(torch.clamp(tensor_data, -clip_obs, clip_obs))

def store_buffer(buffer, data_queue):
    for _ in range(data_queue.qsize()):
        buffer.push(data_queue.get(block = True))

# for Loss calculate
def get_action(actors_model, obs_norm, g_norm):
    batch_size = obs_norm.shape[0]
    acts = torch.zeros(batch_size, n_agents, env_params.dim_action).to(device)
    hands = torch.zeros(batch_size, n_agents, dim_hand).to(device)
    for i in range(n_agents):
        sb_norm = obs_norm[:, i, :]
        input_tensor = torch.cat([sb_norm, g_norm], dim=1)
        act, hand = actors_model[i](input_tensor)
        acts[:, i, :] = act
        hands[:, i, :] = hand
    return acts.view(batch_size, -1), hands.view(batch_size, -1)

def get_value(critics, obs_norm, g_norm, acts_tensor, hands_tensor):
    batch_size = obs_norm.shape[0]
    obs_norm = obs_norm.reshape(batch_size, -1) 
    input_tensor = torch.cat([obs_norm, g_norm], dim=1)
    q_value = critics(input_tensor, acts_tensor, hands_tensor)
    return q_value

def update_network(
        model,
        o_normalizer,
        g_normalizer,
        transitions,
        actor_optimizer,
        critic_optimizer
    ):
    # pre-process the observation and goal
    obs_norm = preproc(o_normalizer, transitions['obs'])
    g_norm = preproc(g_normalizer, transitions['g'])
    obs_next_norm = preproc(o_normalizer, transitions['next_obs'])
    g_next_norm = preproc(g_normalizer, transitions['g'])
    acts_tensor, hands_tensor, r_tensor = transitions['acts'],transitions['hands'], transitions['reward']
    
    for agent in range(n_agents):
        with torch.no_grad():
            # calculate the target Q value function
            acts_next_tensor, hands_next_tensor = get_action(model.actors_target, obs_next_norm, g_next_norm)
            q_next_value = get_value(model.critics_target[agent], obs_next_norm, g_next_norm, acts_next_tensor, hands_next_tensor)
            target_q_value = r_tensor + gamma * q_next_value
            # clip the q value
            clip_return = 1 / (1 - gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = get_value(model.critics[agent], obs_norm, g_norm, acts_tensor, hands_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        acts_real_tensor, hands_real_tensor = get_action(model.actors, obs_norm, g_norm)
        actor_loss = -get_value(model.critics[agent], obs_norm, g_norm, acts_real_tensor, hands_real_tensor).mean()
        # actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()

        # start to update the network
        actor_optimizer[agent].zero_grad()
        actor_loss.backward()
        actor_optimizer[agent].step()

        # update the critic_network
        critic_optimizer[agent].zero_grad()
        critic_loss.backward()
        critic_optimizer[agent].step()

    return actor_loss, critic_loss

def init_demo_buffer(fileName, buffer):
    ''' wait to test '''
    demo_data = np.load(fileName, allow_pickle=True)
    store_item = ['obs', 'ag', 'g', 'acts', 'hands', 'next_obs', 'next_ag', 'reward']
    store_data = [np.array(demo_data[key]) for key in store_item]
    # demo_buffer.push(store_data)
    buffer.push(store_data)

def learn(model_path, data_queue, evalue_queue, actor_queues, logger):
    # initialize function here
    learner_model = Net(env_params, device)
    buffer = replay_buffer(env_params, train_params, logger)
    actor_optimizer = [Adam(x.parameters(), lr= train_params.lr_actor) for x in learner_model.actors]
    critic_optimizer = [Adam(x.parameters(), lr= train_params.lr_critic) for x in learner_model.critics]
    Actor_loss, Critic_loss = 0, 0
    savetime = 0
    if train_params.add_demo:
        init_demo_buffer(train_params.demo_name, buffer)
        logger.info(f'initialize the replay buffer with demo data, replay buffer current size {buffer.current_size}')
    # transfer data
    for queue in actor_queues:
        queue.put({
            'actor_dict' : [deepcopy(actor).cpu().state_dict() for actor in learner_model.actors],
            'normalizer' : {
                'o_mean' : buffer.o_norm.np_mean.copy(),
                'o_std' : buffer.o_norm.np_std.copy(),
                'g_mean' : buffer.g_norm.np_mean.copy(),
                'g_std' : buffer.g_norm.np_std.copy(),
            }
            })
    # waiting buffer data before training
    while buffer.current_size < batch_size:
        store_buffer(buffer, data_queue)
        logger.info(f'wating for samples... buffer current size {buffer.current_size}')
        time.sleep(15)
    
    for step in range(1, train_params.learner_step):
        # self._update_network_gail() # if need Imitation Learning  
                # sample the episodes
        transitions = buffer.sample(batch_size)
        training_data = update_network(
            learner_model,
            buffer.o_norm.normalize_obs,
            buffer.g_norm.normalize_g,
            transitions,
            actor_optimizer,
            critic_optimizer
        )
        Actor_loss += training_data[0]
        Critic_loss += training_data[1]
        # soft update
        def soft_update_target_network(target, source):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1 - polyak) * source_param.data + polyak * target_param.data)

        if step % train_params.update_tar_interval == 0:
            logger.info(f'cur step: {step}')
            for i in range(n_agents):
                soft_update_target_network(learner_model.actors_target[i], learner_model.actors[i])
                soft_update_target_network(learner_model.critics_target[i], learner_model.critics[i])
        # start to do the evaluation
        if step % evalue_interval == 0:
            store_buffer(buffer, data_queue)
            Actor_loss /= evalue_interval
            Critic_loss /= evalue_interval
            logger.info(f'cur step: {step}, actor loss:{Actor_loss:.4f}, critic loss:{Critic_loss:.4f}')
            model_params = {
                'actor_dict' : [deepcopy(actor).cpu().state_dict() for actor in learner_model.actors],
                'normalizer' : {
                    'o_mean' : buffer.o_norm.np_mean.copy(),
                    'o_std' : buffer.o_norm.np_std.copy(),
                    'g_mean' : buffer.g_norm.np_mean.copy(),
                    'g_std' : buffer.g_norm.np_std.copy(),
                }
            }
            for queue in actor_queues:
                queue.put(model_params)
            evalue_params = deepcopy(model_params)
            evalue_params.update(
                { 
                    'step' : step,
                    'actor_loss' : Actor_loss.item() ,
                    'critic_loss' : Critic_loss.item()
                }
            )
            # synchronize the normalizer and actor_worker model not too frequency
            evalue_queue.put(evalue_params)
            # save model
            torch.save([buffer.o_norm.np_mean, buffer.o_norm.np_std, buffer.g_norm.np_mean, buffer.g_norm.np_std,
                    learner_model.actors[0].state_dict(), learner_model.actors[1].state_dict(),
                    learner_model.critics[0].state_dict(), learner_model.critics[1].state_dict()],#, self.disc.state_dict()],
                    model_path + '/' + str(train_params.seed) + '_' +  str(savetime) + '_model.pt')
            savetime += 1
            Actor_loss, Critic_loss = 0, 0
        # if epoch >= 80:
        #     self.Is_train_discrim = False
        #     self.theta *= 0.9  # annealing



#     # def _update_network_gail(self):
#     #     # sample the episodes
#     #     demo_transitions = self.demo_buffer.sample(self.batch_size)
#     #     policy_transitions = self.buffer.sample(self.batch_size)
#     #     obs_demo_norm, g_demo_norm, obs_demo_next_norm, g_demo_next_norm,\
#     #      acts_demo_tensor, hands_demo_tensor, r_tensor = self._preproc(demo_transitions)
#     #     obs_policy_norm, g_policy_norm, obs_next_policy_norm, g_next_policy_norm,\
#     #      acts_policy_tensor, hands_policy_tensor, r_policy_tensor = self._preproc(policy_transitions)
#     #     r_tensor = (1.0 - self.theta) * r_policy_tensor
#     #     batch_size = obs_demo_norm.shape[0]
#     #     states_policy_tensor = num_to_tensor(np.concatenate((obs_policy_norm.reshape(batch_size, -1), g_policy_norm), axis=-1))  # (200,48)
#     #     # demo data
#     #     states_demo_tensor = num_to_tensor(np.concatenate((obs_demo_norm.reshape(batch_size, -1), g_demo_norm), axis=-1))
#     #     if self.Is_train_discrim:
#     #         # train the Discriminator
#     #         self.expert_acc, self.learner_acc = self.train_discrim(states_policy_tensor, states_demo_tensor,
#     #                             acts_policy_tensor, acts_demo_tensor, hands_policy_tensor, hands_demo_tensor)

#     #     for agent in range(self.n_agents):
#     #         with th.no_grad():
#     #             # calculate the target Q value functionS
#     #             r_tensor += self.theta * th.log(self.disc(states_policy_tensor, acts_policy_tensor, hands_policy_tensor))
#     #             acts_next_tensor, hands_next_tensor = self._get_action(obs_next_policy_norm, g_next_policy_norm, target=True)
#     #             q_next_value = self._get_values(agent, obs_next_policy_norm, g_next_policy_norm, acts_next_tensor, hands_next_tensor, target=True)
#     #             q_next_value = q_next_value.detach()
#     #             target_q_value = r_tensor + self.gamma * q_next_value
#     #             target_q_value = target_q_value.detach()
#     #             # clip the q value
#     #             clip_return = 1 / (1 - self.gamma)
#     #             target_q_value = th.clamp(target_q_value, -clip_return, 0)  # 将input各元素限制[-clip_return, 0],得一个新张量

#     #         # the q loss
#     #         real_q_value = self._get_values(agent, obs_policy_norm, g_policy_norm, acts_policy_tensor, hands_policy_tensor, target=False)
#     #         critic_loss = (target_q_value - real_q_value).pow(2).mean()

#     #         # the actor loss
#     #         acts_real_tensor, hands_real_tensor = self._get_action(obs_policy_norm, g_policy_norm, target=False)
#     #         actor_loss = -self._get_values(agent, obs_policy_norm, g_policy_norm, acts_real_tensor, hands_real_tensor, target=False).mean()
#     #         # actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()

#     #         # start to update the network
#     #         self.actor_optimizer[agent].zero_grad()
#     #         actor_loss.backward()
#     #         self.actor_optimizer[agent].step()
#     #         # update the critic_network
#     #         self.critic_optimizer[agent].zero_grad()
#     #         critic_loss.backward()
#     #         self.critic_optimizer[agent].step()

#     # def train_discrim(self, states_policy, states_demo, acts_policy, acts_demo, hands_policy, hands_demo):
#     #     criterion = th.nn.BCELoss()
#     #     for _ in range(5):
#     #         learner = self.disc(states_policy, acts_policy, hands_policy)
#     #         expert = self.disc(states_demo, acts_demo, hands_demo)
#     #         disc_loss = criterion(learner, th.zeros((states_policy.shape[0], 1))) + \
#     #                        criterion(expert, th.ones((states_demo.shape[0], 1)))  # discriminator aims to
#     #         self.disc_optimizer.zero_grad()
#     #         disc_loss.backward()
#     #         self.disc_optimizer.step()
#     #     # learner_acc = ((self.disc(states_policy, action_policy) < 0.5).float()).mean()
#     #     # expert_acc = ((self.disc(states_demo, action_demo) > 0.5).float()).mean()
#     #     learner_acc = (self.disc(states_policy, acts_policy, hands_policy).float()).mean()
#     #     expert_acc = (self.disc(states_demo, acts_demo, hands_demo)).float().mean()
#     #     # print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
#     #     return expert_acc, learner_acc
