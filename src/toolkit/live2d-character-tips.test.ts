import { expect, it } from "vitest";
import tips from "../../public/live2d-models/waifu-tips.json";
import { createLive2DCharacterTips } from "./live2d-character-tips";

it("配置和闲聊数组被缓存后，所有角色仍读取各自台词", () => {
  let id = "0";
  const runtime = createLive2DCharacterTips(tips, () => id);
  const message = runtime.message;
  const idle = message.default;
  for (const next of ["0", "1", "2", "0"] as const) {
    id = next;
    expect([...idle]).toEqual(tips.characterTips[next].message.default);
    for (const key of [
      "hoverBody",
      "tapBody",
      "changeSuccess",
      "changeFail",
      "photo",
      "goodbye",
    ] as const) {
      expect(message[key]).toEqual(tips.characterTips[next].message[key]);
    }
    expect(runtime.mouseover.find((rule) => rule.selector === "a[href='/about/']")?.text).toEqual(
      tips.characterTips[next].mouseover.find((rule) => rule.selector === "a[href='/about/']")
        ?.text,
    );
  }
});

it("角色未定义的字段和选择器回退通用配置，不修改原始 JSON", () => {
  const before = JSON.stringify(tips);
  let id = "2";
  const runtime = createLive2DCharacterTips(tips, () => id);
  expect(runtime.message.hitokoto).toEqual(tips.message.hitokoto);
  expect(runtime.mouseover.find((rule) => rule.selector === "img")).toEqual(
    tips.mouseover.find((rule) => rule.selector === "img"),
  );
  const idle = runtime.message.default;
  idle.push("节日问候");
  id = "1";
  expect([...idle]).toEqual([...tips.characterTips["1"].message.default, "节日问候"]);
  id = "unknown";
  expect(runtime.message.hoverBody).toEqual(tips.message.hoverBody);
  expect(JSON.stringify(tips)).toBe(before);
});
