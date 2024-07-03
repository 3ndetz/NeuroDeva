def dif(long, short):
    for i in range(len(short)):
        if short[i] == long[i]:
            continue
        else:
            return long[i]
    return long[-1]

def mistakes_analyzer(errors: "list[tuple[str, str]]"):
    """Analyze mistakes made by model.
    :param errors - list of (model_ans, correct_ans)
    :returns (loose_symbol, error_symbols, undefined)
            loose_symbol - {symbol: loos_count}
            error_symbols - {correct_symbol: {model_answer: count}} (count=int)
            undefined - list of other mistakes list[tuple[str, str]]
    """
    loose_symbol = {}
    error_symbols = {}
    undefined = []
    for ans, correct_ans in errors:
        if len(ans) != len(correct_ans):
            # if we skip a character
            if len(correct_ans) == len(ans)+1:
                loosed_sym = dif(correct_ans, ans)
                if loosed_sym not in loose_symbol:
                    loose_symbol[loosed_sym] = 0
                loose_symbol[loosed_sym] += 1
            else:
                undefined.append((ans, correct_ans))
            continue
        for sym_correct, sym_fail in zip(correct_ans, ans):
            if sym_correct != sym_fail:
                # if we recognize it incorrectly
                if sym_correct not in error_symbols: error_symbols[sym_correct] = {}
                if sym_fail not in error_symbols[sym_correct]: error_symbols[sym_correct][sym_fail] = 0
                error_symbols[sym_correct][sym_fail] += 1
    return loose_symbol, error_symbols, undefined

