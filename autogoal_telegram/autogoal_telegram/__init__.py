import asyncio
import time
import textwrap

from autogoal.search import Logger
from telegram.ext import Updater, CommandHandler
from telegram import Bot
import re

class TelegramLogger(Logger):
    def __init__(self, token, channel: str = None, name=""):
        self.name = name
        self.channel = int(channel) if channel and channel.isdigit() else channel
        self.last_time = time.time()
        self.last_time_other = time.time()
        self.updater = Updater(token)
        self.dispatcher = self.updater.dispatcher
        self.progress = 0
        self.generations = 1
        self.bests = []
        self.bests_pipelines = []
        self.current = ""
        self.message = self.dispatcher.bot.send_message(
            chat_id=self.channel,
            text=f"<b>{self.name}</b>\nStarting...",
            parse_mode="HTML",
        )
        self.last_message = self.dispatcher.bot.send_message(
            chat_id=self.channel,
            text=f"<b>{self.name} currently:</b>\nStarting...",
            parse_mode="HTML",
        )

    def begin(self, generations, pop_size):
        self.generations = generations
        self._send()
        
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
        self._send()

    def end(self, best_solutions, best_fns):
        self.bests = best_fns
        self._send()
        
    def sample_solution(self, solution):
        text = f"""
        Evaluating pipeline:
        Pipeline: <code>{repr(solution)}</code>
        """
        self._send_update(textwrap.dedent(text))
        
    def error(self, e: Exception, solution):
        text = f"""
        Error evaluating pipeline:
        Pipeline: <code>{repr(solution)}</code>
        Error: <u>{e}</u>
        """
        self._send_update(textwrap.dedent(text))
        
    def eval_solution(self, solution, fitness):
        self.progress += 1
        self._send()

    def _send_update(self, text):
        if not self.channel:
            return
        
        if time.time() - self.last_time_other < 5:
            return

        self.last_time_other = time.time()
        
        try:
            self.last_message.edit_text(
                text=f"<b>{self.name} currently:</b>\n{text}",
                parse_mode="HTML",
            )
        except Exception as e:
            pass

    def _send(self):
        if not self.channel:
            return

        if time.time() - self.last_time < 5:
            return

        self.last_time = time.time()
        pareto_front = "["
        for i in range(len(self.bests_pipelines)):
            pareto_front += "\n---------------\n<code>"
            pareto_front += repr(self.bests_pipelines[i])
            pareto_front += "</code>\n"
            pareto_front += f"macro F1=<code>{self.bests[i][0]}</code>, RAM usage=<code>{self.bests[i][1]}</code>"
            pareto_front += "\n---------------\n"
        pareto_front += "]"
        
        text = textwrap.dedent(
            f"""
            <b>{self.name}</b>
            Iterations: `{self.progress}/{self.generations}`
            Best fitness: `{self.bests}`
            Pareto Front: `{pareto_front}`
            """
        )
        try:
            self.message.edit_text(
                text=text,
                parse_mode="HTML",
            )
        except Exception as e:
            pass
