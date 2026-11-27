import { addons } from 'storybook/manager-api';

// The stories have no args or actions, so the addon panel would stay empty.
addons.setConfig({ showPanel: false });
