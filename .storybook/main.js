/** Storybook for the local workbench pages. Synthetic fixtures only: no run data, frames or videos. */
const config = {
  framework: { name: '@storybook/html-vite', options: {} },
  stories: ['../storybook/stories/**/*.stories.js'],
  core: { disableTelemetry: true },
};
export default config;
