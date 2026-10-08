interface TipRule {
  selector: string;
  text: string | string[];
}

interface CharacterTips {
  message?: Record<string, string | string[]>;
  mouseover?: TipRule[];
  click?: TipRule[];
}

interface WidgetTips extends CharacterTips {
  characterTips?: Record<string, CharacterTips>;
}

/** 库缓存配置和闲聊数组，使用动态读取避免切换角色后仍沿用首次加载的台词。 */
export function createLive2DCharacterTips<T extends WidgetTips>(
  tips: T,
  characterId: () => string,
  welcome?: () => string,
): T {
  const character = () => tips.characterTips?.[characterId()];
  const seasonal: string[] = [];
  const defaults = () => {
    const value = character()?.message?.default ?? tips.message?.default ?? [];
    return [...(Array.isArray(value) ? value : [value]), ...seasonal];
  };
  // CDN 在初始化时缓存 message.default，并向其中追加节日台词。
  const idle = new Proxy<string[]>([], {
    get(target, key, receiver) {
      if (key === "push") return (...items: string[]) => seasonal.push(...items);
      if (key === "length") return defaults().length;
      if (typeof key === "string" && /^\d+$/.test(key)) return defaults()[Number(key)];
      return Reflect.get(target, key, receiver);
    },
  });
  const message = new Proxy(tips.message ?? {}, {
    get(target, key) {
      if (key === "default") return idle;
      if (key === "welcome" && welcome) return welcome();
      return character()?.message?.[String(key)] ?? Reflect.get(target, key);
    },
  });
  const rules = (key: "mouseover" | "click") => {
    const specific = character()?.[key] ?? [];
    const selectors = new Set(specific.map((rule) => rule.selector));
    return [...specific, ...(tips[key] ?? []).filter((rule) => !selectors.has(rule.selector))];
  };
  return new Proxy(tips, {
    get(target, key, receiver) {
      if (key === "message") return message;
      if (key === "mouseover" || key === "click") return rules(key);
      return Reflect.get(target, key, receiver);
    },
  });
}
