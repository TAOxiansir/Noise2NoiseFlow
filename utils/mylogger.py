import logging

def add_logging_level(levelName, levelNum, methodName=None):#levelname:新级别的名称（如“TRACE”）；levelnum：新级别的数值（如 5，要避免与现有级别冲突）methodName: 对应的方法名（如 "trace"，默认为 levelName.lower()）
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.#在 `logging` 模块以及当前配置的日志类中全面新增了一个新的日志类。

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.#`levelName` 成为 `logging` 模块的一个属性，其值为 `levelNum` 。'methodName' 成为 `logging` 本身以及由 `logging.getLoggerClass()` 返回的类（通常为 `logging.Logger`）的一个便捷方法。
    如果未指定 'methodName' ，则使用 `levelName.lower()` 。

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present#为避免意外覆盖现有的属性，如果级别名称已经是 `logging` 模块的属性，或者如果方法名称已经存在于该模块中，
    此方法将会引发一个 `AttributeError` 异常。

    Example
    -------
    >>> add_logging_level('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()
    #防止覆盖已有属性：
    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def log_for_level(self, message, *args, **kwargs):#为logger类创建实例方法
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):#为logging模块创建模块级函数
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)#注册至logging模块中
    setattr(logging, levelName, levelNum)#在代码运行时，给对象“动态地”设置或修改一个属性
    setattr(logging.getLoggerClass(), methodName, log_for_level)
    setattr(logging, methodName, log_to_root)
