import asyncio
import time
import textwrap

from autogoal.search import Logger
from telegram.ext import Updater, CommandHandler
from telegram import Bot


class TelegramLogger(Logger):
    def __init__(self, token, channel: str = None, name=""):
        self.name = name
        self.channel = int(channel) if channel and channel.isdigit() else channel
        self.last_time = time.time()
        self.updater = Updater(token)
        self.dispatcher = self.updater.dispatcher
        self.progress = 0
        self.generations = 1
        self.bests = []
        self.current = ""
        self.message = self.message = self.dispatcher.bot.send_message(
            chat_id=self.channel,
            text=f"**{self.name}**\nStarting...",
            parse_mode="Markdown",
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
        self._send()

    def end(self, best_solutions, best_fns):
        self.bests = best_fns
        self._send()
        
        
    def eval_solution(self, solution, fitness):
        self.progress += 1
        self._send()

    def _send(self):
        if not self.channel:
            return

        if time.time() - self.last_time < 10:
            return

        self.last_time = time.time()

        text = textwrap.dedent(
            f"""
            **{self.name}**
            Pareto Front: `{self.bests}`
            Iterations: `{self.progress}/{self.generations}`
            """
        )
        try:
            self.message.edit_text(
                text=text,
                parse_mode="Markdown",
            )
        except:
            pass
