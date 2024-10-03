const equityUtils = require('./openpokertools.com/src/utils/equity_utils');

const calculateEquity = (heroHand, villainRange, board) => {
  try {
    const [winPercentage, tiePercentage] = equityUtils.calculateHandRangeEquity(
      heroHand, 
      villainRange, 
      board
    );

    console.log(JSON.stringify({
      win: winPercentage,
      tie: tiePercentage
    }));
  } catch (err) {
    console.error("Error calculating equity:", err);
  }
};

const args = process.argv.slice(2);
const heroHand = JSON.parse(args[0]);
const villainRange = JSON.parse(args[1]);
const board = JSON.parse(args[2]);

calculateEquity(heroHand, villainRange, board);