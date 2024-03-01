import asyncio
import time
import textwrap

from autogoal.search import Logger
from telegram.ext import Updater, CommandHandler
from telegram import Bot
import re

class TelegramLogger(Logger):
    def __init__(self, token, channel: str = None, name="", objectives=None):
        self.name = name
        self.channel = int(channel) if channel and channel.isdigit() else channel
        self.last_time = time.time()
        self.last_time_other = time.time()
        self.updater = Updater(token)
        self.dispatcher = self.updater.dispatcher
        self.progress = 0
        
        self.errors = 0
        self.timeout_errors = 0
        self.cuda_errors = 0
        self.memory_errors = 0
        
        self.generations = 1
        self.current_generation = 1
        self.bests = []
        self.bests_pipelines = []
        self.current = ""
        self.message_index = 0
        self.message = self.dispatcher.bot.send_message(
            chat_id=self.channel,
            text=f"<b>{self.name}</b>\nStarting...",
            parse_mode="HTML",
        )
        self.last_solution_message = None
        self.objectives = objectives
        self.message.pin()

    def begin(self, generations, pop_size):
        self.generations = generations
        self._send_status()
        
    def update_best(
        self,
        solution,
        fn,
        new_best_solutions,
        best_solutions,
        new_best_fns,
        best_fns,
        new_dominated_solutions,
    ):
        self.bests = new_best_fns
        self.bests_pipelines = new_best_solutions
        
        self._send_status()

    def end(self, best_solutions, best_fns):
        self.bests_pipelines = best_solutions
        self.bests = best_fns
        self._send_status()
        
    def sample_solution(self, solution):
        text = f"""
        Evaluating pipeline:
        Pipeline: <code>{repr(solution)}</code>
        """
        self.last_solution_message = self._send_new(textwrap.dedent(text))
        
    def error(self, e: Exception, solution):
        text = f"""
        Error:
        <u>{e}</u>
        """
        self.errors += 1
        if re.search("time for execution", str(e).lower()):
            self.timeout_errors += 1
            
        if re.search("cuda out of memory", str(e).lower()):
            self.cuda_errors += 1
        
        if self.last_solution_message:
            previous_message = self.last_solution_message.text_html
            self._send_update(textwrap.dedent(previous_message + text), self.last_solution_message)
        else:
            self._send_new(textwrap.dedent(text))
        
    def eval_solution(self, solution, fitness):
        self.progress += 1
        self._send_status()
        
        if ("inf" in str(fitness).lower()):
            return
            
        
        text = f"""
        Success:
        <u>{fitness}</u>
        """
        
        if self.last_solution_message:
            previous_message = self.last_solution_message.text_html
            self._send_update(textwrap.dedent(previous_message + text), self.last_solution_message)
        else:
            self._send_new(textwrap.dedent(text))

    def _send_update(self, text, message=None):
        if not self.channel:
            return
        
        if message is None:
            message = self.message
        
        if time.time() - self.last_time_other < 2:
            time.sleep(2)

        self.last_time_other = time.time()
        
        try:
            message.edit_text(
                text=f"{text}",
                parse_mode="HTML",
            )
        except Exception as e:
            print(f"Telegram Logger exception: {e}")
            pass
        
    def _send_new(self, text):
        if not self.channel:
            return
        
        if time.time() - self.last_time_other < 2:
            time.sleep(2)

        try:
            self.message_index += 1
            return self.dispatcher.bot.send_message(
                chat_id=self.channel,
                text=f"<b>{self.name} [{self.message_index}]:</b>\n\n{text}",
                parse_mode="HTML",
            )
        except Exception as e:
            print(f"Telegram Logger exception: {e}")
            pass

    def _send_status(self):
        if not self.channel:
            return

        if time.time() - self.last_time < 5:
            time.sleep(5)

        self.last_time = time.time()
        pareto_front = "["
        for i in range(len(self.bests_pipelines)):
            
            eval_text=""
            if not self.objectives:
                eval_text = f"({self.bests[i][0]}, {self.bests[i][1]})"
            else:
                initial = True
                for j in range(len(self.objectives)):
                    obj_name = ""
                    unit = ""
                    
                    if not initial:
                        eval_text += ", "
                    else:
                        initial = False
                    
                    if isinstance(self.objectives[j], tuple):
                        obj_name = self.objectives[j][0]
                        unit = self.objectives[j][1]
                    elif isinstance(self.objectives[j], str):
                        obj_name = self.objectives[j]
                        
                    eval_text += f"{obj_name}=<code>{self.bests[i][j]}"
                    eval_text += f" {unit}" if unit else ""
                    eval_text += "</code>"
            
            pareto_front += "\n---------------\n<code>"
            pareto_front += repr(self.bests_pipelines[i])
            pareto_front += "</code>\n"
            pareto_front += eval_text
            pareto_front += "\n---------------\n"
        pareto_front += "]"
        
        text = textwrap.dedent(
            f"""
            <b>{self.name}</b>
            
            Iterations: `{self.progress}/{self.generations}`
            
            ------ ERRORS ------
            Timeouts: `{self.timeout_errors}`
            CUDA: `{self.cuda_errors}`
            Other: `{self.errors - self.timeout_errors - self.cuda_errors}`
            Total Errors: `{self.errors}`

            ------ PARETO FRONT ------
            `{pareto_front}`
            """
        )
        try:
            self.message.edit_text(
                text=text,
                parse_mode="HTML",
            )
        except Exception as e:
            pass
