import type { NavItemType } from "../navbar/NavTypes";

interface SidebarMenuProps {
  menu?: NavItemType[];
}

interface MenuEntry {
  data: NavItemType;
  isDropdown: boolean;
}

interface MenuEntry {
  data: NavItemType;
  isDropdown: boolean;
}

const getMenuIcon = (item: NavItemType) => item?.icon ?? "";

const isExternalLink = (href: string) => href.startsWith("http://") || href.startsWith("https://");

const linkTarget = (href: string) => (isExternalLink(href) ? "_blank" : undefined);

const linkRel = (href: string) => (isExternalLink(href) ? "noopener noreferrer" : "section");

const ICON_CLASS =
  "ic inline-flex justify-center items-center text-xl flex-wrap align-text-bottom mr-2.5";

const renderNavItems = (items: NavItemType[]): MenuEntry[] =>
  items.map((item) => ({
    data: item,
    isDropdown: Boolean(item.dropbox?.enable && (item.dropbox?.items?.length ?? 0) > 0),
  }));

function SidebarMenu(props: SidebarMenuProps) {
  const menuItems = () => renderNavItems(props.menu ?? []);

  return (
    <nav class="menu p-5 m-0 bg-transparent" aria-label="侧栏菜单导航">
      <ul class="menu-list list-none m-0 p-0">
        {menuItems().map((item) => {
          const icon = item.data.icon;
          const text = item.data.text;
          const url = item.data.href;
          const dropboxItems = item.data.dropbox?.items || [];

          if (item.isDropdown && dropboxItems.length > 0) {
            return (
              <li class="item dropdown">
                <a href={url} target={linkTarget(url)} rel={linkRel(url)}>
                  {icon && <div class={`ic ${icon} ${ICON_CLASS}`}></div>}
                  {text}
                </a>
                <ul class="submenu">
                  {dropboxItems.map((subItem) => (
                    <li class="item">
                      <a
                        href={subItem.href}
                        target={linkTarget(subItem.href)}
                        rel={linkRel(subItem.href)}
                      >
                        {subItem.icon && (
                          <div class={`ic ${getMenuIcon(subItem)} ${ICON_CLASS}`}></div>
                        )}
                        {subItem.text}
                      </a>
                    </li>
                  ))}
                </ul>
              </li>
            );
          }
          return (
            <li class="item">
              <a href={url} target={linkTarget(url)} rel={linkRel(url)}>
                {icon && <div class={`ic ${icon} ${ICON_CLASS}`}></div>}
                {text}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export default SidebarMenu;
