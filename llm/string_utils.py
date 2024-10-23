import random


def add_with_limit(repeatingDict, value, key):
    if key in repeatingDict:
        repeatedList = repeatingDict[key]
        if len(repeatedList) > 1000:
            repeatedList.pop(0)
        if value in repeatedList:
            repeatedList.remove(value)
        repeatedList.append(value)

    else:
        repeatingDict[key] = [value]


class NonRepeatRandom():
    def __init__(self, repeatingDict):
        self.repeatingDict = repeatingDict

    def comment_shuffle_symbols(self, inp: str) -> str:
        inp = inp.replace('#', '[!РЕШЕТКА]')
        inp = inp.replace('@', '[!СОБАКА]')
        return inp
    def uncomment_shuffle_symbols(self, inp: str) -> str:
        inp = inp.replace('[!РЕШЕТКА]', '#')
        inp = inp.replace('[!СОБАКА]', '@')
        return inp

    def apply_shuffle(self, inp: str) -> str:

        formsprocessingB = inp.split('#')  # разбиваем на # и между ними перетусовываем слова рандомным образом
        # sprints(formsprocessingB)
        for i, cho in enumerate(formsprocessingB):
            if i % 2 != 0:
                formsprocessing = formsprocessingB[i].split(' ')
                random.shuffle(formsprocessing)
                formsprocessingB[i] = " ".join(formsprocessing)
        # sprints(formsprocessingB)
        inp = "".join(formsprocessingB)
        inp = inp.replace('#', '')  # перетусовочный символ
        inp = inp.replace('@', ' ')  # заменяет пробел где не нужна перетусовка пробелами
        inp = self.uncomment_shuffle_symbols(inp)
        return inp

    def r(self, values_str: str = None, values_list: list = None, key: str = "default") -> str:
        if values_str is not None:
            rmas = values_str.split(",")
        else:
            rmas = values_list
        # print('DEBUG rmas rpdict',rmas,self.repeatingDict)
        result = None
        repeatedList = self.repeatingDict.get(key, None)
        if repeatedList and len(rmas) > 1:
            # last_found_i = -1
            repeat_found_count = 0
            # print(repeatedList)

            # list_without_repeats = list(set(rmas)-set(repeatedList)) # представляем как множества (все элем -
            # уникальны). Отнимаем от множества А множество Б. с поддержкой дубликатов
            list_without_repeats = [item for item in rmas if item not in repeatedList]

            if list_without_repeats:
                rmas = list_without_repeats
            else:
                accepted_end_element_id = 1
                # надо вычленить только те элементы, которые совпадают
                list_without_repeats = [item for item in repeatedList if item in rmas]
                # print('lwr',list_without_repeats)
                if len(list_without_repeats) > 1:
                    accepted_end_element_id = len(list_without_repeats) // 2
                rmas = list_without_repeats[0:accepted_end_element_id]
        # print(rmas)
        result = random.choice(rmas)
        add_with_limit(self.repeatingDict, result, key)
        return result


if __name__ == '__main__':
    nrr = NonRepeatRandom({})

    def s(x): return nrr.apply_shuffle(x)

    def r(x): return nrr.r(x)

    print(
        f"""как дела {r("1,2,3,4")} {r("3,2,3,4")} {r("1,2,3,4")} {r("5,2,3,4")} {r("1,2,6,4")} {r("1,2,3,4")} {r("3,2,3,4")} {r("1,2,3,4")} {r("5,2,3,4")} {r("1,2,6,4")} {r("1,2,3,4")} {r("3,2,3,4")} {r("1,2,3,4")} {r("5,2,3,4")} {r("1,2,6,4")}  """)
    print(
        s(
            f"""\n\n#как дела# {r("да,нет")}@{r("да,нет")} #ты кто {r("да,нет")} {r("да,нет")} {r("да,нет")}# ### 352jk523 ### 2###"""))
