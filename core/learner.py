import torch
import os
import sys
import time
import numpy as np
from torch.optim import Adam
from core.buffer import replay_buffer
from copy import deepcopy

class Learner:
    def __init__(self, env_params, train_params, actor_model, model, data_queue, evalue_queue, logger):
        for key, val in train_params.items():
            exec(f'self.{key} = "{val}"') if type(val) == str else exec(f'self.{key} = {val}' )
        for key, val in env_params.items():
            exec(f'self.{key} = "{val}"') if type(val) == str else exec(f'self.{key} = {val}' )
        self.env_params = env_params
        self.train_params = train_params
        self.device = train_params.device
        self.model = model
        self.actor_model = actor_model
        self.data_queue = data_queue
        self.evalue_queue = evalue_queue
        self.logger = logger
        self.actor_optimizer = [Adam(x.parameters(), lr=self.lr_actor) for x in self.model.actors]
        self.critic_optimizer = [Adam(x.parameters(), lr=self.lr_critic) for x in self.model.critics]
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # path to save the model
        self.model_path = os.path.join(self.save_dir, self.env_name)
        self.savetime = 0 
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def reset_buffer(self):
        self.buffer = replay_buffer(self.env_params, self.train_params, logger= self.logger)
        self.o_norm = self.buffer.o_norm
        self.g_norm = self.buffer.g_norm
        return self.buffer

    def learn(self):
        while self.buffer.current_size < self.batch_size:
            self.store_buffer()
            self.logger.info(f'wating for samples... buffer current size {self.buffer.current_size}')
            time.sleep(5)
        self.Actor_loss = 0
        self.Critic_loss = 0
        for step in range(1, self.learner_step):
            # self._update_network_gail() # if need Imitation Learning  
                    # sample the episodes
            transitions = self.buffer.sample(self.batch_size)
            for key, val in transitions.items():
                transitions[key].to(self.device)
            training_data = self._update_network(transitions)
            self.Actor_loss += training_data[0]
            self.Critic_loss += training_data[1]
            # soft update
            if step % self.update_tar_interval == 0:
                self.logger.info(f'cur step: {step}')
                for i in range(self.n_agents):
                    self._soft_update_target_network(self.model.actors_target[i], self.model.actors[i])
                    self._soft_update_target_network(self.model.critics_target[i], self.model.critics[i])
            # start to do the evaluation
            if step % self.evalue_interval == 0:
                self.store_buffer()
                self.Actor_loss /= self.evalue_interval
                self.Critic_loss /= self.evalue_interval
                self.logger.info(f'cur step: {step}, \
                actor loss:{self.Actor_loss:.4f}, \
                critic loss:{self.Critic_loss:.4f}')
                self.evalue_queue.put(
                    {  
                        'step' : step,
                        'o_norm_mean' : self.o_norm.np_mean.copy(),
                        'o_norm_std' : self.o_norm.np_std.copy(),
                        'g_norm_mean' : self.g_norm.np_mean.copy(),
                        'g_norm_std' : self.g_norm.np_std.copy(),
                        'actors' : deepcopy(self.model.actors),
                        'actor_loss' : self.Actor_loss.item() ,
                        'critic_loss' : self.Critic_loss.item()
                    }
                )
                self.Actor_loss, self.Critic_loss = 0, 0
                # synchronize the normalizer and actor_worker model not too frequency
                self.actor_model.update(self.model)
                # save model
                torch.save([self.o_norm.np_mean, self.o_norm.np_std, self.g_norm.np_mean, self.g_norm.np_std,
                     self.model.actors[0].state_dict(), self.model.actors[1].state_dict(),
                     self.model.critics[0].state_dict(), self.model.critics[1].state_dict()],#, self.disc.state_dict()],
                     self.model_path + '/' + str(self.seed) + '_' + str(self.add_demo) +
                     str(self.savetime) + '_model.pt')
                self.savetime += 1
            # if epoch >= 80:
            #     self.Is_train_discrim = False
            #     self.theta *= 0.9  # annealing
    
    def store_buffer(self):
        cur_size = self.data_queue.qsize()
        for _ in range(cur_size):
            self.buffer.push(self.data_queue.get(block = True))
    # choose action for the agent for exploration or evaluation

    # update the network
    def _update_network(self, transitions):
        # pre-process the observation and goal
        obs_norm, g_norm, obs_next_norm, g_next_norm, acts_tensor, hands_tensor, r_tensor = self._preproc(transitions)
        for agent in range(self.n_agents):
            with torch.no_grad():
                # calculate the target Q value function
                acts_next_tensor, hands_next_tensor = self._get_action(obs_next_norm, g_next_norm, target=True)
                q_next_value = self._get_values(agent, obs_next_norm, g_next_norm, acts_next_tensor, hands_next_tensor, target=True)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.gamma * q_next_value
                target_q_value = target_q_value.detach()
                # clip the q value
                clip_return = 1 / (1 - self.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)  # 将input各元素限制[-clip_return, 0],得一个新张量
            # the q loss
            real_q_value = self._get_values(agent, obs_norm, g_norm, acts_tensor, hands_tensor, target=False)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            # the actor loss
            acts_real_tensor, hands_real_tensor = self._get_action(obs_norm, g_norm, target=False)
            actor_loss = -self._get_values(agent, obs_norm, g_norm, acts_real_tensor, hands_real_tensor, target=False).mean()
            # actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()

            # start to update the network
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[agent].step()

            # update the critic_network
            self.critic_optimizer[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[agent].step()
        return actor_loss, critic_loss

    def _preproc(self, transitions):
        o, o_next, g = transitions['obs'], transitions['next_obs'], transitions['g']
        o = torch.clamp(o, -self.clip_obs, self.clip_obs)  # self.args.clip_obs = 200
        g = torch.clamp(g, -self.clip_obs, self.clip_obs)
        next_o = torch.clamp(o_next, -self.clip_obs, self.clip_obs)  # self.args.clip_obs = 200
        next_g = torch.clamp(g, -self.clip_obs, self.clip_obs)
        # start to do the update
        obs_norm = self.o_norm.normalize_obs(o)
        g_norm = self.g_norm.normalize_g(g)
        obs_next_norm = self.o_norm.normalize_obs(next_o)
        g_next_norm = self.g_norm.normalize_g(next_g)
        acts_tensor,hands_tensor, r_tensor = transitions['acts'],transitions['hands'], transitions['reward']
        return obs_norm, g_norm, obs_next_norm, g_next_norm, acts_tensor, hands_tensor, r_tensor

    # for Loss calculate
    def _get_action(self, obs_norm, g_norm, target):
        batch_size = obs_norm.shape[0]
        acts = torch.zeros(batch_size, self.n_agents, self.dim_action)
        hands = torch.zeros(batch_size, self.n_agents, self.dim_hand)
        for i in range(self.n_agents):
            sb_norm = obs_norm[:, i, :]
            input_tensor = torch.cat([sb_norm, g_norm], dim=1)
            if target:
                act, hand = self.model.actors_target[i](input_tensor)
            else:
                act, hand = self.model.actors[i](input_tensor)
            acts[:, i, :] = act
            hands[:, i, :] = hand
        return acts.view(batch_size, -1), hands.view(batch_size, -1)

    def _get_values(self, agent, obs_norm, g_norm, acts_tensor, hands_tensor, target):
        batch_size = obs_norm.shape[0]
        obs_norm = obs_norm.reshape(batch_size, -1) 
        input_tensor = torch.cat([obs_norm, g_norm], dim=1)
        if target:
            q_value = self.model.critics_target[agent](input_tensor, acts_tensor, hands_tensor)
        else:
            q_value = self.model.critics[agent](input_tensor, acts_tensor, hands_tensor)
        return q_value

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * source_param.data + self.polyak * target_param.data)

    # def _update_network_gail(self):
    #     # sample the episodes
    #     demo_transitions = self.demo_buffer.sample(self.batch_size)
    #     policy_transitions = self.buffer.sample(self.batch_size)
    #     obs_demo_norm, g_demo_norm, obs_demo_next_norm, g_demo_next_norm,\
    #      acts_demo_tensor, hands_demo_tensor, r_tensor = self._preproc(demo_transitions)
    #     obs_policy_norm, g_policy_norm, obs_next_policy_norm, g_next_policy_norm,\
    #      acts_policy_tensor, hands_policy_tensor, r_policy_tensor = self._preproc(policy_transitions)
    #     r_tensor = (1.0 - self.theta) * r_policy_tensor
    #     batch_size = obs_demo_norm.shape[0]
    #     states_policy_tensor = num_to_tensor(np.concatenate((obs_policy_norm.reshape(batch_size, -1), g_policy_norm), axis=-1))  # (200,48)
    #     # demo data
    #     states_demo_tensor = num_to_tensor(np.concatenate((obs_demo_norm.reshape(batch_size, -1), g_demo_norm), axis=-1))
    #     if self.Is_train_discrim:
    #         # train the Discriminator
    #         self.expert_acc, self.learner_acc = self.train_discrim(states_policy_tensor, states_demo_tensor,
    #                             acts_policy_tensor, acts_demo_tensor, hands_policy_tensor, hands_demo_tensor)

    #     for agent in range(self.n_agents):
    #         with th.no_grad():
    #             # calculate the target Q value functionS
    #             r_tensor += self.theta * th.log(self.disc(states_policy_tensor, acts_policy_tensor, hands_policy_tensor))
    #             acts_next_tensor, hands_next_tensor = self._get_action(obs_next_policy_norm, g_next_policy_norm, target=True)
    #             q_next_value = self._get_values(agent, obs_next_policy_norm, g_next_policy_norm, acts_next_tensor, hands_next_tensor, target=True)
    #             q_next_value = q_next_value.detach()
    #             target_q_value = r_tensor + self.gamma * q_next_value
    #             target_q_value = target_q_value.detach()
    #             # clip the q value
    #             clip_return = 1 / (1 - self.gamma)
    #             target_q_value = th.clamp(target_q_value, -clip_return, 0)  # 将input各元素限制[-clip_return, 0],得一个新张量

    #         # the q loss
    #         real_q_value = self._get_values(agent, obs_policy_norm, g_policy_norm, acts_policy_tensor, hands_policy_tensor, target=False)
    #         critic_loss = (target_q_value - real_q_value).pow(2).mean()

    #         # the actor loss
    #         acts_real_tensor, hands_real_tensor = self._get_action(obs_policy_norm, g_policy_norm, target=False)
    #         actor_loss = -self._get_values(agent, obs_policy_norm, g_policy_norm, acts_real_tensor, hands_real_tensor, target=False).mean()
    #         # actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()

    #         # start to update the network
    #         self.actor_optimizer[agent].zero_grad()
    #         actor_loss.backward()
    #         self.actor_optimizer[agent].step()
    #         # update the critic_network
    #         self.critic_optimizer[agent].zero_grad()
    #         critic_loss.backward()
    #         self.critic_optimizer[agent].step()

    # def train_discrim(self, states_policy, states_demo, acts_policy, acts_demo, hands_policy, hands_demo):
    #     criterion = th.nn.BCELoss()
    #     for _ in range(5):
    #         learner = self.disc(states_policy, acts_policy, hands_policy)
    #         expert = self.disc(states_demo, acts_demo, hands_demo)
    #         disc_loss = criterion(learner, th.zeros((states_policy.shape[0], 1))) + \
    #                        criterion(expert, th.ones((states_demo.shape[0], 1)))  # discriminator aims to
    #         self.disc_optimizer.zero_grad()
    #         disc_loss.backward()
    #         self.disc_optimizer.step()
    #     # learner_acc = ((self.disc(states_policy, action_policy) < 0.5).float()).mean()
    #     # expert_acc = ((self.disc(states_demo, action_demo) > 0.5).float()).mean()
    #     learner_acc = (self.disc(states_policy, acts_policy, hands_policy).float()).mean()
    #     expert_acc = (self.disc(states_demo, acts_demo, hands_demo)).float().mean()
    #     # print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
    #     return expert_acc, learner_acc
